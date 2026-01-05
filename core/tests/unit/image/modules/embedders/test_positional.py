"""
Tests for FluxPosEmbed module.
"""
from torch import randn

from components.tranformers.modules.embedders.positional import PositionalEmbedder


class TestFluxPosEmbed:
    """Test suite for FluxPosEmbed module."""

    def test_initialization(self):
        """Test basic initialization."""
        theta = 10000
        axes_dim = [64, 64]

        embedder = PositionalEmbedder(theta, axes_dim)

        assert embedder.theta == theta
        assert embedder.axes_dimension == axes_dim

    def test_forward__basic(self):
        """Test basic forward pass."""
        theta = 10000
        axes_dim = [64, 64]
        seq_len = 16
        n_axes = 2

        embedder = PositionalEmbedder(theta, axes_dim)
        ids = randn(seq_len, n_axes)

        freqs_cos, freqs_sin = embedder(ids)

        assert freqs_cos is not None and freqs_cos.numel() > 0
        assert freqs_sin is not None and freqs_sin.numel() > 0
