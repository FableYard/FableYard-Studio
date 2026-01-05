"""
Tests for get_timestep_embedding function.
"""
from torch import arange

from components.tranformers.modules.embedders.get_timestep_embedding import get_timestep_embedding


class TestGetTimestepEmbedding:
    """Test suite for get_timestep_embedding function."""

    def test_get_timestep_embedding(self):
        """Test basic timestep embedding generation."""
        batch_size = 4
        embedding_dim = 128

        timesteps = arange(batch_size).float()
        embeddings = get_timestep_embedding(timesteps, embedding_dim)

        assert embeddings is not None
        assert embeddings.numel() > 0
