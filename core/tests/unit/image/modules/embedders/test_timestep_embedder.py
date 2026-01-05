"""
Tests for TimestepEmbedding module.
"""
from torch import randn

from components.tranformers.modules.embedders.timestep_embedder import TimestepEmbedding


class TestTimestepEmbedding:
    """Test suite for TimestepEmbedding module."""

    def test_initialization(self):
        """Test basic initialization."""
        in_channels = 256
        time_embed_dim = 512

        embedder = TimestepEmbedding(in_channels, time_embed_dim)

        assert embedder.linear_1 is not None
        assert embedder.linear_2 is not None
        assert embedder.act is not None
        assert embedder.cond_proj is None
        assert embedder.post_act is None

    def test_initialization__with_post_act(self):
        """Test initialization with post-activation."""
        in_channels = 256
        time_embed_dim = 512

        embedder = TimestepEmbedding(in_channels, time_embed_dim, post_act_fn="gelu")

        assert embedder.post_act is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size = 4
        in_channels = 256
        time_embed_dim = 512

        embedder = TimestepEmbedding(in_channels, time_embed_dim)
        sample = randn(batch_size, in_channels)

        output = embedder(sample)

        assert output is not None
        assert output.numel() > 0

    def test_forward__with_out_dim(self):
        """Test forward pass with custom output dimension."""
        batch_size = 4
        in_channels = 256
        time_embed_dim = 512
        out_dim = 1024

        embedder = TimestepEmbedding(in_channels, time_embed_dim, output_channel_count=out_dim)
        sample = randn(batch_size, in_channels)

        output = embedder(sample)

        assert output.shape == (batch_size, out_dim)

    def test_forward__with_condition(self):
        """Test forward pass with conditional input."""
        batch_size = 4
        in_channels = 256
        time_embed_dim = 512
        cond_proj_dim = 768

        embedder = TimestepEmbedding(in_channels, time_embed_dim, cond_proj_dim=cond_proj_dim)
        sample = randn(batch_size, in_channels)
        condition = randn(batch_size, cond_proj_dim)

        output = embedder(sample, condition)

        assert output.shape == (batch_size, time_embed_dim)
