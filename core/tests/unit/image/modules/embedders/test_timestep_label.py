"""
Tests for CombinedTimestepLabelEmbeddings module.
"""
from torch import all, isfinite, randn, randint

from components.tranformers.modules.embedders.timestep_label import CombinedTimestepLabelEmbeddings


class TestCombinedTimestepLabelEmbeddings:
    """Test suite for CombinedTimestepLabelEmbeddings module."""

    def test_initialization(self):
        """Test basic initialization."""
        num_classes = 10
        embedding_dim = 512
        class_dropout_prob = 0.1

        embedder = CombinedTimestepLabelEmbeddings(num_classes, embedding_dim, class_dropout_prob)

        assert embedder.time_proj is not None
        assert embedder.timestep_embedder is not None
        assert embedder.class_embedder is not None

    def test_forward(self):
        """Test basic forward pass."""
        batch_size = 4
        num_classes = 10
        embedding_dim = 512

        embedder = CombinedTimestepLabelEmbeddings(num_classes, embedding_dim, class_dropout_prob=0.0)
        embedder.eval()

        timestep = randn(batch_size)
        class_labels = randint(0, num_classes, (batch_size,))

        output = embedder(timestep, class_labels)

        assert output is not None
        assert output.numel() > 0

    # TODO: Investigate why this test provides coverage
    def test_forward__training_mode(self):
        """Test forward pass in training mode with dropout."""
        batch_size = 4
        num_classes = 10
        embedding_dim = 512

        embedder = CombinedTimestepLabelEmbeddings(num_classes, embedding_dim, class_dropout_prob=0.5)
        embedder.train()

        timestep = randn(batch_size)
        class_labels = randint(0, num_classes, (batch_size,))

        output = embedder(timestep, class_labels)

        assert output.shape == (batch_size, embedding_dim)
        assert all(isfinite(output))
