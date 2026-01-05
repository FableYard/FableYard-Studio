"""
Tests for LabelEmbedding module.
"""
from torch import all, isfinite, randint, zeros

from components.tranformers.modules.embedders.label import LabelEmbedding


class TestLabelEmbedding:
    """Test suite for LabelEmbedding module."""

    def test_initialization__basic(self):
        """Test basic initialization."""
        num_classes = 10
        hidden_size = 128
        dropout_prob = 0.1

        embedder = LabelEmbedding(num_classes, hidden_size, dropout_prob)

        assert embedder.class_count == num_classes
        assert embedder.dropout_prob == dropout_prob
        assert embedder.embedding_table is not None

    def test_forward__basic(self):
        """Test basic forward pass."""
        batch_size = 4
        num_classes = 10
        hidden_size = 128
        dropout_prob = 0.0

        embedder = LabelEmbedding(num_classes, hidden_size, dropout_prob)
        embedder.eval()  # Set to eval mode to disable dropout

        labels = randint(0, num_classes, (batch_size,))
        embeddings = embedder(labels)

        assert embeddings is not None
        assert embeddings.numel() > 0

    def test_forward__training_with_dropout(self):
        """Test forward pass in training mode with dropout."""
        batch_size = 4
        num_classes = 10
        hidden_size = 128
        dropout_prob = 0.5

        embedder = LabelEmbedding(num_classes, hidden_size, dropout_prob)
        embedder.train()  # Set to training mode

        labels = randint(0, num_classes, (batch_size,))
        embeddings = embedder(labels)

        assert embeddings.shape == (batch_size, hidden_size)
        assert all(isfinite(embeddings))

    def test_forward__force_drop_ids(self):
        """Test forward pass with forced drop ids."""
        batch_size = 4
        num_classes = 10
        hidden_size = 128
        dropout_prob = 0.1

        embedder = LabelEmbedding(num_classes, hidden_size, dropout_prob)
        embedder.eval()

        labels = randint(0, num_classes, (batch_size,))
        force_drop_ids = zeros(batch_size)
        force_drop_ids[0] = 1  # Force drop first label

        embeddings = embedder(labels, force_drop_ids=force_drop_ids)

        assert embeddings.shape == (batch_size, hidden_size)
        assert all(isfinite(embeddings))
