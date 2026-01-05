# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Z-Image Pipeline
Handles text-to-image generation using Z-Image Turbo model with Qwen3 text encoder.
"""
import logging
from gc import collect
from pathlib import Path

import torch

from components import KullbackLeibler
from components.qwen2_tokenizer import Qwen2Tokenizer
from components.qwen3_text_encoder import Qwen3TextEncoder
from components.schedulers.flowmatcheulerdiscrete import FlowMatchEulerDiscrete
from components.tranformers.z_image.turbo_transformer import ZImageTransformer
from utils import ImageSaver
from utils.logger import info


class ZImagePipeline:
    """
    Z-Image text-to-image pipeline.
    Uses Qwen3 text encoder with chat templates for prompt processing.
    """

    def __init__(
        self,
        model_path: Path,
        batch_size: int,
        prompts: dict[str, dict[str, str]],
        step_count: int,
        image_height: int,
        image_width: int,
        seed: int,
        guidance_scale: float,
        image_name: str | None
    ):
        self.batch_size = batch_size
        self.positive_prompt = prompts['qwen']['positive']
        self.negative_prompt = prompts['qwen'].get('negative', '')  # Default to empty string if not provided
        self.step_count = step_count
        self.image_height = image_height
        self.image_width = image_width
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.image_name = image_name

        # Component paths
        self.tokenizer_path = Path(model_path, "tokenizer")
        self.text_encoder_path = Path(model_path, "text_encoder")
        self.scheduler_path = Path(model_path, "scheduler")
        self.transformer_path = Path(model_path, "transformer")
        self.vae_path = Path(model_path, "vae")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"Running on device: {self.device}")

    def execute(self):
        # ========================================================================
        # 1. Initialize Text Encoder and Tokenizer
        # ========================================================================
        info(f"Initializing Qwen3 text encoder and tokenizer...")
        tokenizer = Qwen2Tokenizer(self.tokenizer_path, self.device)
        text_encoder = Qwen3TextEncoder(self.text_encoder_path, self.device)

        # Apply chat template to prompt
        logging.info(f"Positive prompt: {repr(self.positive_prompt)}")
        logging.info(f"Negative prompt: {repr(self.negative_prompt)}")
        messages = [
            {"role": "user", "content": self.positive_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # Tokenize prompt
        max_sequence_length = 512
        text_inputs = tokenizer.encode(
            formatted_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device).bool()

        # Encode to embeddings (use second-to-last hidden state)
        encoder_hidden_states = text_encoder.encode(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract embeddings for valid tokens only (matching reference implementation)
        embeddings_list = []
        for i in range(encoder_hidden_states.shape[0]):  # Iterate over batch dimension
            embeddings_list.append(encoder_hidden_states[i][attention_mask[i]])

        # Release tokenizer and encoder
        del tokenizer, text_encoder, text_inputs, text_input_ids, attention_mask
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ========================================================================
        # 2. Generate Latents First (needed for scheduler setup)
        # ========================================================================
        info(f"Calculating latent dimensions...")
        # Z-Image latent calculation: 2 * (height // (vae_scale * 2))
        # VAE upscales by 8x, so for 512x512 output we need 64x64 latents
        vae_scale_factor = 8
        latent_height = 2 * (self.image_height // (vae_scale_factor * 2))
        latent_width = 2 * (self.image_width // (vae_scale_factor * 2))
        latent_channels = 16

        # ========================================================================
        # 3. Initialize Scheduler
        # ========================================================================
        info(f"Initializing scheduler...")

        scheduler = FlowMatchEulerDiscrete(self.scheduler_path, device=self.device)

        # Calculate image sequence length for scheduler based on latent dimensions
        image_seq_len = latent_height * latent_width

        # Set timesteps (shift calculation is done internally by scheduler)
        scheduler.set_timesteps(
            step_count=self.step_count,
            image_sequence_length=image_seq_len
        )
        timesteps = scheduler._timesteps

        # ========================================================================
        # 4. Initialize Transformer
        # ========================================================================
        info(f"Initializing Z-Image transformer...")
        transformer = ZImageTransformer(self.transformer_path, self.device)

        # ========================================================================
        # 5. Generate Initial Latents
        # ========================================================================
        info(f"Generating initial latents...")

        # Create generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        latents = torch.randn(
            (self.batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # ========================================================================
        # 6. Denoising Loop
        # ========================================================================
        info(f"Starting denoising loop ({self.step_count} steps)...")

        # Get transformer dtype once before the loop
        transformer_dtype = next(transformer.parameters()).dtype

        for i, t in enumerate(timesteps):
            info(f"Step {i + 1}/{self.step_count} - Timestep: {t.item():.4f}")

            # Use no_grad to prevent gradient accumulation
            with torch.no_grad():
                # Normalize timestep to [0, 1]
                timestep = (1000 - t) / 1000
                timestep_expanded = timestep.expand(latents.shape[0])

                # Prepare latent input - transformer expects a list of (C, F, H, W) tensors (one per batch item)
                latent_model_input = latents.to(transformer_dtype)
                # Add frame dimension and split batch into list (matching reference pipeline)
                latent_model_input = latent_model_input.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
                latent_list = list(latent_model_input.unbind(0))  # [(C, 1, H, W), ...]

                # Run transformer (returns tuple with list inside)
                output = transformer(
                    latent_list,  # List of tensors, one per batch item
                    timestep_expanded,
                    embeddings_list,  # Already a list from text encoding
                )

                # Extract noise prediction from output (tuple[list[Tensor]])
                # output is (list[Tensor],) where each tensor is (C, 1, H, W)
                noise_pred_list = output[0]

                # Stack back to batch dimension: [(C, 1, H, W), ...] -> (B, C, 1, H, W)
                noise_pred = torch.stack(noise_pred_list, dim=0)

                # Remove frame dimension: (B, C, 1, H, W) -> (B, C, H, W)
                noise_pred = noise_pred.squeeze(2)

                # Apply guidance if needed
                if self.guidance_scale > 1.0:
                    # For classifier-free guidance, you would need negative prompts
                    # For now, using unconditional guidance is not implemented
                    pass

                # Scheduler step
                noise_pred = -noise_pred.float()
                latents = scheduler.step(noise_pred, latents)

                # Clean up intermediate tensors
                del latent_model_input, latent_list, output, noise_pred_list, noise_pred

            # Free GPU memory periodically
            if (i + 1) % 2 == 0:
                torch.cuda.empty_cache()

        # Release transformer
        del transformer, scheduler, embeddings_list
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ========================================================================
        # 7. Decode with VAE
        # ========================================================================
        info(f"Decoding latents with VAE...")
        vae = KullbackLeibler(self.vae_path, self.device)
        vae.load()  # Load VAE weights

        # Scale latents for VAE (latents are already 4D: B, C, H, W)
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(vae_dtype)
        latents = (latents / vae._scaling_factor) + vae._shift_factor

        # Decode
        decoded = vae.decode(latents)

        # Release VAE
        del vae
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ========================================================================
        # 8. Post-process and Save
        # ========================================================================
        info(f"Post-processing and saving images...")
        # VAE already returns decoded images in (B, C, H, W) format - no unpatchify needed
        images = decoded

        # Save to PROJECT_ROOT/user/output/ (not core/outputs/)
        # z_image.py -> z_image/ -> txt2img/ -> pipelines/ -> src/ -> core/ -> PROJECT_ROOT
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        output_dir = project_root / "user" / "output"
        image_saver = ImageSaver(output_dir=output_dir)

        # If batch size > 1, save the first image from the batch
        if images.shape[0] > 1:
            image_to_save = images[0]
        else:
            image_to_save = images

        # Use image_name if provided, otherwise use default
        filename = f"{self.image_name}.png" if self.image_name else "example.png"
        result_path = image_saver.save(image_to_save, filename=filename)

        info(f"Image saved to: {result_path}")
        return str(result_path)
