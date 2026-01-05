# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from components.tranformers.modules.embedders.apply_rotary_emb import apply_rotary_emb
from components.tranformers.modules.attendants.utils import _get_qkv_projections
from components.tranformers.modules.fused_ops import fused_qkv_prep_single, fused_qkv_prep_dual


class AttentionProcessor:
    """
    Processor for Attention that uses PyTorch's optimized scaled_dot_product_attention.
    This automatically leverages Flash Attention 2 or Memory-Efficient Attention when available.
    """

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn,
            hidden_states,
            encoder_hidden_states
        )
        assert not query.isnan().any(), f"NaN in query after projection"
        assert not key.isnan().any(), f"NaN in key after projection"
        assert not value.isnan().any(), f"NaN in value after projection"

        # Use fused QKV preparation for memory bandwidth optimization
        if attn.added_kv_proj_dim is not None:
            # Dual-stream: fuse unflatten + transpose + norm + concatenation
            query, key, value = fused_qkv_prep_dual(
                query, key, value,
                encoder_query, encoder_key, encoder_value,
                attn.heads,
                attn.norm_q, attn.norm_k,
                attn.norm_added_q, attn.norm_added_k
            )
        else:
            # Single-stream: fuse unflatten + transpose + norm
            query, key, value = fused_qkv_prep_single(
                query, key, value,
                attn.heads,
                attn.norm_q, attn.norm_k
            )

        assert not query.isnan().any(), f"NaN in query after fused prep"
        assert not key.isnan().any(), f"NaN in key after fused prep"
        assert not value.isnan().any(), f"NaN in value after fused prep"

        # Ensure contiguous memory layout for optimal attention performance
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)
            assert not query.isnan().any(), f"NaN in query after rotary emb"
            assert not key.isnan().any(), f"NaN in key after rotary emb"

        # Tensors are already in [B, H, S, D] format for scaled_dot_product_attention
        # Use PyTorch's optimized scaled_dot_product_attention
        # This automatically uses Flash Attention 2 or Memory-Efficient Attention when available
        hidden_states = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        assert not hidden_states.isnan().any(), f"NaN in hidden_states after scaled_dot_product_attention"

        # Transpose back from [B, H, S, D] to [B, S, H, D] and flatten
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        assert not hidden_states.isnan().any(), f"NaN in hidden_states after transpose/flatten"

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            assert not hidden_states.isnan().any(), f"NaN in hidden_states after split"
            assert not encoder_hidden_states.isnan().any(), f"NaN in encoder_hidden_states after split"

            hidden_states = attn.to_out[0](hidden_states)
            assert not hidden_states.isnan().any(), f"NaN in hidden_states after to_out[0]"

            hidden_states = attn.to_out[1](hidden_states)
            assert not hidden_states.isnan().any(), f"NaN in hidden_states after to_out[1]"

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            assert not encoder_hidden_states.isnan().any(), f"NaN in encoder_hidden_states after to_add_out"

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
