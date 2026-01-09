# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Transformer Component
FLUX transformer model with pipeline lifecycle management.
"""
import json
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import torch
from torch import Tensor, LongTensor, cat
import gc

try:
    from torch import is_torch_npu_available
except ImportError:
    def is_torch_npu_available():
        return False
from torch.nn import Linear, ModuleList, Module

from core.src.components.tranformers.modules.embedders.guided_timestep_embedder import CombinedTimestepGuidanceTextProjEmbeddings
from core.src.components.tranformers.modules.normalizers.adaptive_layer_continuous import AdaLayerNormContinuous
from core.src.components.tranformers.flux.modules.blocks import FluxTransformerBlock
from core.src.components.tranformers.flux.modules.blocks.single import SingleStreamBlock
from core.src.components.tranformers.modules.embedders.positional import PositionalEmbedder


class FluxTransformer(Module):
    """
    FLUX Transformer component with flow matching architecture.

    Combines transformer model implementation with pipeline lifecycle management.
    Implements dual-stream (separate image/text processing) and single-stream
    (joint processing) architecture for diffusion-based image generation.
    """
    def __init__(
        self,
        component_path: Path | str,
        device: str,
        adapters: Optional[dict],
    ):
        Module.__init__(self)
        self.component_path = Path(component_path)
        self.device = device
        self.adapters = adapters

        config = json.load(open(self.component_path / "config.json"))
        self.patch_size: int = config['patch_size']
        self.in_channels: int = config['in_channels']
        self.layer_count: int = config['num_layers']
        self.single_layer_count: int = config['num_single_layers']
        self.attention_head_dimensions: int = config['attention_head_dim']
        self.attention_heads_count: int = config['num_attention_heads']
        self.joint_attention_dimensions: int = config['joint_attention_dim']
        self.pooled_projection_dimensions: int = config['pooled_projection_dim']
        self.axes_dimensions_rope: list = [16, 56, 56]

        self._out_channels: int = self.in_channels
        self._inner_dimensions: int = self.attention_heads_count * config['attention_head_dim']

    def load(self) -> None:
        """Load FLUX transformer architecture and weights"""
        # Initialize transformer architecture on meta device (no memory allocation)
        with torch.device("meta"):
            # Position embeddings
            self.positional_embeddings = PositionalEmbedder(theta=10000, axes_dimension=self.axes_dimensions_rope)

            # Time, text, and guidance embeddings (match checkpoint naming)
            text_time_guidance_cls = CombinedTimestepGuidanceTextProjEmbeddings
            self.time_text_embed = text_time_guidance_cls(
                embedding_dim=self._inner_dimensions,
                pooled_projection_dim=self.pooled_projection_dimensions
            )

            # Input embedders
            self.context_embedder = Linear(self.joint_attention_dimensions, self._inner_dimensions)
            self.x_embedder = Linear(self.in_channels, self._inner_dimensions)

            # Dual-stream transformer blocks (separate image and text processing)
            self.transformer_blocks = ModuleList(
                [
                    FluxTransformerBlock(
                        dim=self._inner_dimensions,
                        num_attention_heads=self.attention_heads_count,
                        attention_head_dim=self.attention_head_dimensions,
                    )
                    for _ in range(self.layer_count)
                ]
            )

            # Single-stream transformer blocks (joint processing)
            self.single_transformer_blocks = ModuleList(
                [
                    SingleStreamBlock(
                        dim=self._inner_dimensions,
                        num_attention_heads=self.attention_heads_count,
                        attention_head_dim=self.attention_head_dimensions,
                    )
                    for _ in range(self.single_layer_count)
                ]
            )

            # Output layers
            self.norm_out = AdaLayerNormContinuous(self._inner_dimensions, self._inner_dimensions, elementwise_affine=False, eps=1e-6)
            self.proj_out = Linear(self._inner_dimensions, self.patch_size * self.patch_size * self._out_channels, bias=True)

        # Use accelerate for efficient model loading
        from accelerate import load_checkpoint_in_model

        # Model is already on meta device from the context manager above
        # Use accelerate to load checkpoint with proper memory management
        print("Loading model with accelerate...")
        load_checkpoint_in_model(
            self,
            checkpoint=str(self.component_path),
            device_map={"": self.device},  # Load everything to our device
            dtype=torch.bfloat16,  # Convert to bfloat16
            offload_state_dict=True,  # Don't keep full state dict in memory
        )

        print("Model loaded")

        if self.adapters:
            from core.src.components.adapters.adapter_patcher import AdapterPatcher
            # from core.src.components.adapters.mapper import FluxAdapterMapper

            print(f"Patching {len(self.adapters)} adapters into model...")

            # Convert self.adapters (dict[str, path or state dict]) into AdapterPatcher
            patcher = AdapterPatcher(transformer_state_dict=self.state_dict(), model_type="flux")

            # Add each adapter with strength
            for adapter_name, adapter_info in self.adapters.items():
                # adapter_info can be dict: {"path": path, "strength": float}
                adapter_path = adapter_info["path"]
                strength = adapter_info.get("strength", 1.0)
                patcher.add_adapter(adapter_path, strength)

            # Apply all patches
            patcher.apply_patches()

            # Release patcher and adapter data to free memory
            del patcher

            print("Adapters applied successfully.")

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Forward pass
    @torch.inference_mode()
    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        pooled_projections: Tensor = None,
        timestep: LongTensor = None,
        img_ids: Tensor = None,
        txt_ids: Tensor = None,
        guidance: Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, ]:
        """
        Forward pass through the FLUX transformer.

        Implements dual-stream and single-stream processing for flow matching
        diffusion. Processes image latents and text conditioning through
        transformer blocks with rotary position embeddings.

        Args:
            hidden_states: Image latent features
            encoder_hidden_states: Text encoder hidden states (from T5)
            pooled_projections: Pooled text embeddings (from CLIP)
            timestep: Current diffusion timestep
            img_ids: Image position IDs for rotary embeddings
            txt_ids: Text position IDs for rotary embeddings
            guidance: Guidance scale for conditioning
            joint_attention_kwargs: Additional attention parameters

        Returns:
            Tuple containing the denoised output
        """
        # Embed image latents
        hidden_states = self.x_embedder(hidden_states)

        # Scale timestep and guidance (Flux convention: scale by 1000)
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000

        # Create guided timestep conditioning (combines timestep + guidance + pooled text)
        guided_timesteps = self.time_text_embed(timestep, guidance, pooled_projections)

        # Embed text conditioning
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Generate rotary position embeddings
        ids = cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.positional_embeddings(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.positional_embeddings(ids)

        # Dual-stream processing (separate image and text paths)
        for i, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                guided_timesteps=guided_timesteps,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Single-stream processing (joint image + text)
        for i, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                guided_timesteps=guided_timesteps,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Adaptive normalization and output projection
        hidden_states = self.norm_out(hidden_states, guided_timesteps)

        output = self.proj_out(hidden_states)

        return (output,)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_layers={self.layer_count}, "
            f"num_single_layers={self.single_layer_count}, "
            f"inner_dim={self._inner_dimensions})"
        )
