"""
Tests for CombinedTimestepGuidanceTextProjEmbeddings module.
"""
from torch import randn

from components.tranformers.modules.embedders.guided_timestep_embedder import CombinedTimestepGuidanceTextProjEmbeddings


class TestCombinedTimestepGuidanceTextProjEmbeddings:
    """Test suite for CombinedTimestepGuidanceTextProjEmbeddings module."""

    def test_initialization(self):
        """Test basic initialization."""
        embedding_dim = 512
        pooled_projection_dim = 768

        embedder = CombinedTimestepGuidanceTextProjEmbeddings(embedding_dim, pooled_projection_dim)

        assert embedder.time_proj is not None
        assert embedder.timestep_embedder is not None
        assert embedder.guidance_embedder is not None
        assert embedder.text_embedder is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size = 4
        embedding_dim = 512
        pooled_projection_dim = 768

        embedder = CombinedTimestepGuidanceTextProjEmbeddings(embedding_dim, pooled_projection_dim)

        timestep = randn(batch_size)
        guidance = randn(batch_size)
        pooled_projection = randn(batch_size, pooled_projection_dim)

        output = embedder(timestep, guidance, pooled_projection)

        assert output is not None
        assert output.numel() > 0
