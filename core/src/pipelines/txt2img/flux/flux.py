# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from gc import collect
from pathlib import Path
from typing import Optional

import torch

from core.src.components import CLIPTokenizer, T5Tokenizer, T5TextEncoder, CLIPTextEncoder, \
    KullbackLeibler
from core.src.components.schedulers.flowmatcheulerdiscrete import FlowMatchEulerDiscrete
from core.src.components.tranformers.flux.fluxtransformer import FluxTransformer
from core.src.utils import ImageSaver, unpatchify
from core.src.utils.latent_generator import LatentGenerator
from core.src.utils.logger import info


class FluxPipeline:
    def __init__(
            self,
            model_path: Path,
            batch_size: int,
            prompts: dict[str, dict[str, str]],
            adapters: Optional[dict[str, dict[str, str | float]]],
            step_count: int,
            image_height: int,
            image_width: int,
            seed: int,
            guidance_scale: float,
            image_name: Optional[str]
    ):
        self.batch_size = batch_size
        self.clip_prompt = prompts['clip']['positive']
        self.t5_prompt = prompts['t5']['positive']
        self.adapters = adapters
        self.step_count = step_count
        self.image_height = image_height
        self.image_width = image_width
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.image_name = image_name

        self.clip_tokenizer_path = Path(model_path, "tokenizer")
        self.t5_tokenizer_path = Path(model_path, "tokenizer_2")
        self.clip_encoder_path = Path(model_path, "text_encoder")
        self.t5_encoder_path = Path(model_path, "text_encoder_2")
        self.scheduler_path = Path(model_path, "scheduler")
        self.transformer_path = Path(model_path, "transformer")
        self.vae_path = Path(model_path, "vae")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"Running on device: {self.device}")

    def execute(self):
        # ========================================================================
        # 1. Initialize Text Encoders and Tokenizers
        # ========================================================================
        info(f"Initializing text encoders and tokenizers...")
        clip_tokenizer = CLIPTokenizer(self.clip_tokenizer_path, self.device)
        t5_tokenizer = T5Tokenizer(self.t5_tokenizer_path, self.device)
        t5_tokenizer.load()

        clip_encoder = CLIPTextEncoder(self.clip_encoder_path, self.device)
        t5_encoder = T5TextEncoder(self.t5_encoder_path, self.device)

        # Tokenize prompts
        info(f"[DEBUG] CLIP prompt: {repr(self.clip_prompt)}")
        info(f"[DEBUG] T5 prompt: {repr(self.t5_prompt)}")
        clip_tokens = clip_tokenizer.encode(self.clip_prompt, padding=True)
        t5_tokens = t5_tokenizer.encode(self.t5_prompt, padding=True)

        # Release tokenizers
        del clip_tokenizer, t5_tokenizer
        collect()

        # Encode to embeddings
        pooled_projections = clip_encoder.encode_pooled(clip_tokens)
        encoder_hidden_states = t5_encoder.encode(t5_tokens)

        # Generate txt_ids
        text_seq_len = encoder_hidden_states.shape[1]
        txt_ids = torch.zeros((text_seq_len, 3), dtype=torch.long)

        # Release encoders
        del clip_encoder, t5_encoder, clip_tokens, t5_tokens
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ========================================================================
        # 2. Initialize Scheduler
        # ========================================================================
        info(f"Initializing scheduler...")

        latent_generator = LatentGenerator(
            height=self.image_height,
            width=self.image_width,
            vae_downsampling_factor=16,
            latent_channels=16,
            device=self.device,
            dtype=torch.bfloat16,
        )

        scheduler = FlowMatchEulerDiscrete(self.scheduler_path, device=self.device)
        scheduler.set_timesteps(self.step_count, latent_generator.sequence_length)

        # ========================================================================
        # 3. Initialize Transformer
        # ========================================================================
        info(f"Initializing transformer...")

        with torch.device("meta"):
            transformer = FluxTransformer(self.transformer_path, self.device, self.adapters)

        transformer.load()

        # ========================================================================
        # 4. Diffusion Loop
        # ========================================================================
        info(f"Running {self.step_count} steps...")

        latents, img_ids = latent_generator.generate(batch_size=self.batch_size, seed=self.seed)
        transformer.eval()

        latents = latents.to(device=self.device, dtype=torch.bfloat16)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device, dtype=torch.bfloat16)
        pooled_projections = pooled_projections.to(device=self.device, dtype=torch.bfloat16)
        img_ids = img_ids.to(device=self.device, dtype=torch.bfloat16)
        txt_ids = txt_ids.to(device=self.device, dtype=torch.bfloat16)

        guidance_scale = 3.5
        guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.bfloat16)
        guidance = guidance.expand(latents.shape[0])

        with torch.no_grad():
            for i, timestep in enumerate(scheduler._timesteps):
                timestep = timestep.expand(latents.shape[0]).to(device=self.device, dtype=latents.dtype)

                model_output = transformer.forward(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    img_ids=img_ids,
                    txt_ids=txt_ids
                )

                predicted_output = model_output[0]

                latents = scheduler.step(
                    model_output=predicted_output,
                    sample=latents,
                    return_dict=False
                )[0]

        # Release transformer
        del transformer, encoder_hidden_states, pooled_projections, img_ids, txt_ids, guidance
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ========================================================================
        # 5. Unpatchify Latents
        # ========================================================================
        info(f"[5/7] Unpatchifying latents...")

        import math

        seq_len = latents.shape[1]
        patch_grid_size = int(math.sqrt(seq_len))
        sample_size = patch_grid_size * 16

        unpatchified_latents = unpatchify(
            latents=latents,
            height=sample_size,
            width=sample_size,
            vae_scale_factor=16
        )

        # ========================================================================
        # 5. Initialize VAE
        # ========================================================================
        info(f"Initializing VAE...")

        vae = KullbackLeibler(self.vae_path, self.device)
        vae.load()
        vae = vae.to(dtype=torch.float32)

        # Convert latents to match VAE dtype (float32)
        unpatchified_latents = unpatchified_latents.to(dtype=torch.float32)

        with torch.no_grad():
            decoded_output = vae.decode(unpatchified_latents)

        # ========================================================================
        # 6. Save Image
        # ========================================================================
        info(f"Saving image...")

        # Save to PROJECT_ROOT/user/output/ (not core/outputs/)
        # flux.py -> flux/ -> txt2img/ -> pipelines/ -> src/ -> core/ -> PROJECT_ROOT
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        output_dir = project_root / "user" / "output"
        image_saver = ImageSaver(output_dir=output_dir)
        save_path = image_saver.save(decoded_output, filename=f"{self.image_name}.png")

        info(f"Pipeline complete: {save_path}")
        return str(save_path)