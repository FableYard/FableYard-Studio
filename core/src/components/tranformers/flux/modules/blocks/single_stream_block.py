# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, float16
from torch.nn import Module, Linear, GELU

from components.tranformers.modules.normalizers.adaptive_layer_zero_single import AdaLayerNormZeroSingle
from components.tranformers.modules.attendants.attention import Attention
from components.tranformers.flux.modules.processor import AttentionProcessor
from components.tranformers.modules.fused_ops import fused_gated_residual


class SingleStreamBlock(Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = Attention(
            query_dimension=dim,
            dim_head=attention_head_dim,
            head_count=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=AttentionProcessor(),
            eps=1e-6,
            pre_only=True,
        )

        # Cached buffers for concatenation operations (optimization)
        # Reused across forward passes to avoid repeated memory allocations
        self._seq_concat_buffer = None  # For encoder + image sequence concatenation
        self._feat_concat_buffer = None  # For attention + MLP feature concatenation

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        guided_timesteps: Tensor,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        total_seq_len = text_seq_len + img_seq_len
        batch_size = hidden_states.shape[0]

        # Optimized concatenation using pre-allocated buffer
        # Avoids repeated memory allocations (38 blocks × inference steps)
        if (self._seq_concat_buffer is None or
            self._seq_concat_buffer.shape != (batch_size, total_seq_len, self.dim)):
            self._seq_concat_buffer = torch.empty(
                (batch_size, total_seq_len, self.dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

        # In-place copy to buffer (faster than cat)
        self._seq_concat_buffer[:, :text_seq_len] = encoder_hidden_states
        self._seq_concat_buffer[:, text_seq_len:] = hidden_states
        hidden_states = self._seq_concat_buffer

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=guided_timesteps)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Optimized feature concatenation using pre-allocated buffer
        # Combines attention output + MLP output before projection
        attention_dimension = attn_output.shape[2]
        mlp_dimension = mlp_hidden_states.shape[2]
        total_dimension = attention_dimension + mlp_dimension
        sequence_length = attn_output.shape[1]

        if (self._feat_concat_buffer is None or
            self._feat_concat_buffer.shape != (batch_size, sequence_length, total_dimension)):
            self._feat_concat_buffer = torch.empty(
                (batch_size, sequence_length, total_dimension),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

        # In-place copy to buffer (faster than cat)
        self._feat_concat_buffer[:, :, :attention_dimension] = attn_output
        self._feat_concat_buffer[:, :, attention_dimension:] = mlp_hidden_states
        hidden_states = self._feat_concat_buffer

        # Fused gated projection + residual for memory bandwidth optimization
        projection_output = self.proj_out(hidden_states)
        hidden_states = fused_gated_residual(residual, gate, projection_output)

        if hidden_states.dtype == float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

        return encoder_hidden_states, hidden_states
