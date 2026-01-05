# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Tuple, Dict, Any

from torch import Tensor, float16
from torch.nn import Module, LayerNorm

from components.tranformers.modules.feed_forward import FeedForward
from components.tranformers.modules.normalizers.adaptive_layer_zero import AdaLayerNormZero
from components.tranformers.modules.attendants.attention import Attention
from components.tranformers.flux.modules.processor import AttentionProcessor


class DualStreamBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        # qk_norm: str = "rms_norm",
        eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = Attention(
            query_dimension=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            head_count=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=AttentionProcessor(),
            eps=eps,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.ff = FeedForward(input_channel_count=dim, output_channel_count=dim, activation_function="gelu-approximate")
        self.ff_context = FeedForward(input_channel_count=dim, output_channel_count=dim, activation_function="gelu-approximate")

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        guided_timesteps: Tensor,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=guided_timesteps)
        assert not norm_hidden_states.isnan().any(), f"NaN in norm_hidden_states after norm1"

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=guided_timesteps
        )
        assert not norm_encoder_hidden_states.isnan().any(), f"NaN in norm_encoder_hidden_states after norm1_context"

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs
        assert not attn_output.isnan().any(), f"NaN in attn_output after attention"
        assert not context_attn_output.isnan().any(), f"NaN in context_attn_output after attention"

        # Process attendants outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        assert not hidden_states.isnan().any(), f"NaN in hidden_states after attention residual"

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        assert not norm_hidden_states.isnan().any(), f"NaN in norm_hidden_states after norm2"

        ff_output = self.ff(norm_hidden_states)
        assert not ff_output.isnan().any(), f"NaN in ff_output after feedforward"

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        assert not hidden_states.isnan().any(), f"NaN in hidden_states after ff residual"

        # Process attendants outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        assert not encoder_hidden_states.isnan().any(), f"NaN in encoder_hidden_states after context attention residual"

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        assert not norm_encoder_hidden_states.isnan().any(), f"NaN in norm_encoder_hidden_states after norm2_context"

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        assert not context_ff_output.isnan().any(), f"NaN in context_ff_output after ff_context"

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        assert not encoder_hidden_states.isnan().any(), f"NaN in encoder_hidden_states after context ff residual"

        # Clip to prevent overflow/underflow
        if hidden_states.dtype == float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        elif hidden_states.dtype.is_floating_point and hidden_states.dtype != float16:
            # For bfloat16 and other dtypes, use a conservative clip to prevent numerical instability
            hidden_states = hidden_states.clip(-1e4, 1e4)

        if encoder_hidden_states.dtype == float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        elif encoder_hidden_states.dtype.is_floating_point and encoder_hidden_states.dtype != float16:
            encoder_hidden_states = encoder_hidden_states.clip(-1e4, 1e4)

        return encoder_hidden_states, hidden_states
