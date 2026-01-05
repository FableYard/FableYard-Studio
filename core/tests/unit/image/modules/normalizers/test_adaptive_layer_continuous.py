"""
Tests for AdaLayerNormContinuous module.
"""
from torch import all, isfinite, randn

from components.tranformers.modules.normalizers.adaptive_layer_continuous import AdaLayerNormContinuous


class TestAdaLayerNormContinuous:
    """Test suite for AdaLayerNormContinuous normalization layer."""

    def test_initialization__layer_norm(self):
        """Test initialization with layer_norm."""
        embedding_dim = 512
        conditioning_embedding_dim = 768

        norm = AdaLayerNormContinuous(embedding_dim, conditioning_embedding_dim, norm_type="layer_norm")

        assert norm.silu is not None
        assert norm.linear is not None
        assert norm.norm is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, embedding_dim = 4, 16, 512
        conditioning_embedding_dim = 768

        norm = AdaLayerNormContinuous(embedding_dim, conditioning_embedding_dim)
        x = randn(batch_size, seq_len, embedding_dim)
        conditioning_embedding = randn(batch_size, conditioning_embedding_dim)

        output = norm(x, conditioning_embedding)

        assert output is not None
        assert output.numel() > 0

    def test_forward__with_rms_norm(self):
        """Test forward pass with RMS normalization."""
        batch_size, seq_len, embedding_dim = 4, 16, 512
        conditioning_embedding_dim = 768

        norm = AdaLayerNormContinuous(embedding_dim, conditioning_embedding_dim, norm_type="rms_norm")
        x = randn(batch_size, seq_len, embedding_dim)
        conditioning_embedding = randn(batch_size, conditioning_embedding_dim)

        output = norm(x, conditioning_embedding)

        assert output.shape == x.shape
        assert all(isfinite(output))
