"""
Tests for AdaLayerNormZeroSingle module.
"""
from torch import randn

from components.tranformers.modules.normalizers.adaptive_layer_zero_single import AdaLayerNormZeroSingle


class TestAdaLayerNormZeroSingle:
    """Test suite for AdaLayerNormZeroSingle normalization layer."""

    def test_initialization(self):
        """Test basic initialization."""
        embedding_dim = 512

        norm = AdaLayerNormZeroSingle(embedding_dim)

        assert norm.silu is not None
        assert norm.linear is not None
        assert norm.norm is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, embedding_dim = 4, 16, 512

        norm = AdaLayerNormZeroSingle(embedding_dim)
        x = randn(batch_size, seq_len, embedding_dim)
        emb = randn(batch_size, embedding_dim)

        x_out, gate_msa = norm(x, emb)

        assert x_out is not None
        assert gate_msa is not None
        assert x_out.numel() > 0
        assert gate_msa.numel() > 0
