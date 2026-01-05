# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Diffusion Component
Runs the diffusion loop with per-iteration progress events.
"""
from typing import Any, Dict
import torch
from components.base_component import PipelineComponent
from utils.event_emitter import EventEmitter
from utils import info


class DiffusionComponent(PipelineComponent):
    """
    Executes the diffusion denoising loop.
    Emits progress events after each iteration.
    """

    @property
    def component_name(self) -> str:
        return "Diffusion Loop"

    def load(self) -> None:
        """No loading needed - uses transformer from previous component"""
        pass

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run diffusion loop with guidance.

        Expected inputs:
            - latents: Tensor (initial noise)
            - transformer: FluxTransformer
            - scheduler: FlowMatchingEulerDiscrete
            - timesteps: Tensor
            - encoder_hidden_states: Tensor
            - pooled_projections: Tensor
            - img_ids: Tensor
            - txt_ids: Tensor

        Returns:
            - latents: Tensor (denoised latents)
        """
        latents = inputs["latents"]
        transformer = inputs["transformer"]
        scheduler = inputs["scheduler"]
        timesteps = inputs["timesteps"]
        encoder_hidden_states = inputs["encoder_hidden_states"]
        pooled_projections = inputs["pooled_projections"]
        img_ids = inputs["img_ids"]
        txt_ids = inputs["txt_ids"]

        # Move everything to device
        latents = latents.to(device=self.device, dtype=torch.float16)
        encoder_hidden_states = encoder_hidden_states.to(device=self.device, dtype=torch.float16)
        pooled_projections = pooled_projections.to(device=self.device, dtype=torch.float16)
        img_ids = img_ids.to(device=self.device, dtype=torch.float16)
        txt_ids = txt_ids.to(device=self.device, dtype=torch.float16)

        # Guidance scale
        guidance_scale = self.config.get("guidance_scale", 3.5)
        guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.bfloat16)
        guidance = guidance.expand(latents.shape[0])

        # Get EventEmitter for progress updates
        emitter = EventEmitter.get_instance()
        total_steps = len(timesteps)

        # Diffusion loop
        with torch.no_grad():
            for i, timestep in enumerate(timesteps):
                timestep = timestep.expand(latents.shape[0]).to(latents.dtype)

                # Forward pass through transformer
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

                # Scheduler step
                latents = scheduler.step(
                    model_output=predicted_output,
                    timestep=timestep,
                    sample=latents,
                    return_dict=False
                )[0]

                # Log step completion
                info(f"Step {i + 1}/{total_steps} completed")

                # Emit progress event after each iteration mapped to Transformer card
                if emitter.is_active:
                    emitter.emit_diffusion_progress(i + 1, total_steps, component="Transformer")

        # Move intermediate tensors to CPU to free GPU memory before VAE decode
        # Use non_blocking=True for async transfers (GPU can continue working)
        encoder_hidden_states = encoder_hidden_states.to('cpu', non_blocking=True)
        pooled_projections = pooled_projections.to('cpu', non_blocking=True)
        img_ids = img_ids.to('cpu', non_blocking=True)
        txt_ids = txt_ids.to('cpu', non_blocking=True)

        # Explicitly remove transformer and scheduler from inputs to break references
        # This allows garbage collection while they're still in context
        del transformer
        del scheduler

        # Force garbage collection to free the transformer immediately
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "latents": latents
        }

    def cleanup(self) -> None:
        """No cleanup needed - transformer is cleaned by TransformerComponent"""
        super().cleanup()

    def __enter__(self):
        """Override to skip emitting component_start event (no frontend card for this)"""
        # Don't emit start event - this component's progress is shown on Transformer card
        # Just clear cache and load
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Override to skip emitting component_complete event (no frontend card for this)"""
        # Don't emit complete event - handled by Transformer card via diffusion_progress
        self.cleanup()
        return False
