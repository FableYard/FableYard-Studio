# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import gc
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from components.tranformers.z_image.modules.final_layer import FinalLayer
from components.tranformers.z_image.modules.utils import patchify_and_embed, unpatchify, build_unified_sequence
from components.tranformers.z_image.modules.constants import ADALN_EMBED_DIM
from components.tranformers.z_image.modules.timestep_embedder import TimestepEmbedder
from components.tranformers.z_image.modules.rope_embedder import RopeEmbedder
from components.tranformers.z_image.modules.single_stream_block import ZImageTransformerBlock
from components.tranformers.modules.normalizers.root_mean_squared import RMSNorm


class ZImageTransformer(nn.Module):
    def __init__(
            self,
            component_path: Path | str,
            device: str,
    ) -> None:
        super().__init__()
        self.component_path = Path(component_path)
        self.device = device

        # Load config from local directory
        config = json.load(open(self.component_path / "config.json"))

        # Extract config values (using HuggingFace config key names)
        all_patch_size = tuple(config.get('all_patch_size'))
        all_f_patch_size = tuple(config.get('all_f_patch_size'))
        input_channel_count = config.get('in_channels')
        dimension = config.get('dim')
        head_count = config.get('n_heads')
        rope_theta = config.get('rope_theta')
        time_scale = config.get('t_scale')

        self.input_channel_count = input_channel_count
        self.output_channel_count = input_channel_count
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dimension = dimension
        self.head_count = head_count
        self.rope_theta = rope_theta
        self.time_scale = time_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        # Use accelerate for efficient model loading
        from accelerate import init_empty_weights, load_checkpoint_in_model

        # Build architecture on meta device (no memory allocation)
        with init_empty_weights():
            all_x_embedder = {}
            all_final_layer = {}
            for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
                x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * input_channel_count, dimension, bias=True)
                all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

                final_layer = FinalLayer(dimension, patch_size * patch_size * f_patch_size * self.output_channel_count)
                all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

            self.all_x_embedder = nn.ModuleDict(all_x_embedder)
            self.all_final_layer = nn.ModuleDict(all_final_layer)
            self.noise_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        1000 + layer_id,
                        dimension,
                        head_count,
                        config.get('n_kv_heads'),
                        config.get('norm_eps'),
                        config.get('qk_norm'),
                        modulation=True,
                    )
                    for layer_id in range(config.get('n_refiner_layers'))
                ]
            )
            self.context_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        layer_id,
                        dimension,
                        head_count,
                        config.get('n_kv_heads'),
                        config.get('norm_eps'),
                        config.get('qk_norm'),
                        modulation=False,
                    )
                    for layer_id in range(config.get('n_refiner_layers'))
                ]
            )
            self.t_embedder = TimestepEmbedder(min(dimension, ADALN_EMBED_DIM), mid_size=1024)
            self.cap_embedder = nn.Sequential(
                RMSNorm(config.get('cap_feat_dim'), epsilon=config.get('norm_eps')),
                nn.Linear(config.get('cap_feat_dim'), dimension, bias=True)
            )

            self.x_pad_token = nn.Parameter(torch.empty((1, dimension)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dimension)))

            self.layers = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        layer_id,
                        dimension,
                        head_count,
                        config.get('n_kv_heads'),
                        config.get('norm_eps'),
                        config.get('qk_norm')
                    )
                    for layer_id in range(config.get('n_layers'))
                ]
            )

        # Set attributes after layer creation
        head_dim = dimension // head_count
        assert head_dim == sum(config.get('axes_dims'))
        self.axes_dimensions = config.get('axes_dims')
        self.axes_lengths = config.get('axes_lens')

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=self.axes_dimensions, axes_lens=self.axes_lengths)

        # Model is already on meta device from the context manager above
        # Use accelerate to load checkpoint with proper memory management
        print("Loading Z-Image model with accelerate...")

        load_checkpoint_in_model(
            self,
            checkpoint=str(self.component_path),
            device_map={"": self.device},
            dtype=torch.bfloat16,
            offload_state_dict=True,
        )

        print("Z-Image model loading complete!")

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_sequence(
        self,
        features: List[torch.Tensor],
        position_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        device: torch.device = None,
    ):
        """Prepare sequence: apply pad token, RoPE embed, pad to batch, create attention mask."""
        item_sequence_length = [len(f) for f in features]
        max_sequence_length = max(item_sequence_length)
        bsz = len(features)

        # Pad token
        feats_cat = torch.cat(features, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token
        features = list(feats_cat.split(item_sequence_length, dim=0))

        # RoPE
        cis_frequencies = list(self.rope_embedder(torch.cat(position_ids, dim=0)).split([len(p) for p in position_ids], dim=0))

        # Pad to batch
        features = pad_sequence(features, batch_first=True, padding_value=0.0)
        cis_frequencies = pad_sequence(cis_frequencies, batch_first=True, padding_value=0.0)[:, : features.shape[1]]

        # Attention mask
        attention_mask = torch.zeros((bsz, max_sequence_length), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_sequence_length):
            attention_mask[i, :seq_len] = 1

        return features, cis_frequencies, attention_mask, item_sequence_length


    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        device = x[0].device

        # Single timestep embedding for all tokens
        adaln_input = self.t_embedder(t * self.time_scale).type_as(x[0])

        # Patchify (basic mode)
        (x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask) = patchify_and_embed(
            x, cap_feats, patch_size, f_patch_size
        )

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
        x, x_freqs, x_mask, _ = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, device
        )

        for layer in self.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input)

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        cap_feats, cap_freqs, cap_mask, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, device
        )

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs, adaln_input)

        # Unified sequence (basic mode: [x, cap])
        unified, unified_freqs, unified_mask = build_unified_sequence(
            x, x_freqs, x_seqlens, cap_feats, cap_freqs, cap_seqlens, device
        )

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]

        # Final layer
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)

        # Unpatchify
        x = unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, self.output_channel_count)

        return (x,)
