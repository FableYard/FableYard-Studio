"""
Tests for apply_rotary_emb function.
"""
from torch import all, isfinite, randn

from components.tranformers.modules.embedders.apply_rotary_emb import apply_rotary_emb


class TestApplyRotaryEmb:
    """Test suite for apply_rotary_emb function."""

    def test_apply_real__sequence_dim_1(self):
        """Test applying rotary embeddings with use_real=True and sequence_dim=1."""
        batch_size, seq_len, heads, dim = 2, 16, 4, 64

        x = randn(batch_size, seq_len, heads, dim)
        cos = randn(seq_len, dim)
        sin = randn(seq_len, dim)
        freqs_cis = (cos, sin)

        output = apply_rotary_emb(x, freqs_cis, use_real=True, sequence_dim=1)

        assert output.shape == x.shape
        assert all(isfinite(output))

    def test_apply_real__unbind_dim_minus_2(self):
        """Test with use_real_unbind_dim=-2 (stable audio)."""
        batch_size, heads, seq_len, dim = 2, 4, 16, 64

        x = randn(batch_size, heads, seq_len, dim)
        cos = randn(seq_len, dim)
        sin = randn(seq_len, dim)
        freqs_cis = (cos, sin)

        output = apply_rotary_emb(
            x, freqs_cis, use_real=True, use_real_unbind_dim=-2, sequence_dim=2
        )

        assert output.shape == x.shape
        assert all(isfinite(output))
