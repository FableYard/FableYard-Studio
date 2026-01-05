"""
Tests for FP32LayerNorm module.
"""
from torch import randn

from components.tranformers.modules.normalizers.layer_fp32 import FP32LayerNorm


class TestFP32LayerNorm:
    """Test suite for FP32LayerNorm normalization layer."""

    def test_initialization(self):
        """Test basic initialization."""
        normalized_shape = 128

        norm = FP32LayerNorm(normalized_shape)

        assert norm.normalized_shape == (normalized_shape,)
        assert norm.weight is not None
        assert norm.bias is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 4, 16, 128

        norm = FP32LayerNorm(dim)
        hidden_states = randn(batch_size, seq_len, dim)

        output = norm(hidden_states)

        assert output is not None
        assert output.numel() > 0
