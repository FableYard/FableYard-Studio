# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    """
    Compute Q, K, V projections for both main and encoder hidden states.
    Everything stays in bfloat16 for memory efficiency.

    Returns:
        tuple: (query, key, value, encoder_query, encoder_key, encoder_value)
    """
    assert not hidden_states.isnan().any(), f"NaN in hidden_states input to QKV projections"

    # Simple direct projections - no dtype conversions
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    assert not query.isnan().any(), f"NaN in query after to_q projection"
    assert not key.isnan().any(), f"NaN in key after to_k projection"
    assert not value.isnan().any(), f"NaN in value after to_v projection"

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        assert not encoder_hidden_states.isnan().any(), f"NaN in encoder_hidden_states input to QKV projections"

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        assert not encoder_query.isnan().any(), f"NaN in encoder_query after add_q_proj"
        assert not encoder_key.isnan().any(), f"NaN in encoder_key after add_k_proj"
        assert not encoder_value.isnan().any(), f"NaN in encoder_value after add_v_proj"

    return query, key, value, encoder_query, encoder_key, encoder_value
