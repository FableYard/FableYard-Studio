"""
Tests for get_1d_rotary_pos_embed function.
"""
from torch import all, isfinite
from numpy import arange as np_arange

from components.tranformers.modules.embedders.get_1d_rotary_pos_embed import get_1d_rotary_pos_embed


class TestGet1DRotaryPosEmbed:
    """Test suite for get_1d_rotary_pos_embed function."""

    def test_basic_embed__real(self):
        """Test basic rotary embedding generation with use_real=True."""
        dim = 64
        seq_len = 16

        freqs_cos, freqs_sin = get_1d_rotary_pos_embed(dim, seq_len)

        assert freqs_cos is not None and freqs_cos.numel() > 0
        assert freqs_sin is not None and freqs_sin.numel() > 0

    def test_embed_with_numpy_pos(self):
        """Test with numpy array as position input."""
        dim = 64
        pos = np_arange(16)

        freqs_cos, freqs_sin = get_1d_rotary_pos_embed(dim, pos)

        assert freqs_cos.shape == (16, dim)
        assert freqs_sin.shape == (16, dim)
        assert all(isfinite(freqs_cos))
        assert all(isfinite(freqs_sin))
