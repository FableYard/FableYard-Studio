"""
Tests for AdaLayerNormZero module.
"""
from torch import randn, randint

from components.tranformers.modules.normalizers import AdaLayerNormZero


class TestAdaLayerNormZero:
    """Test suite for AdaLayerNormZero normalization layer."""

    def test_init(self):
        """Test initialization with num_embeddings (creates embedding layer)."""
        embedding_dim = 512
        num_embeddings = 10

        norm = AdaLayerNormZero(embedding_dim, num_embeddings)

        assert norm.emb is not None
        assert norm.silu is not None
        assert norm.linear is not None
        assert norm.norm is not None

    def test_initialization__fp32_layer_norm(self):
        """Test initialization with fp32_layer_norm."""
        embedding_dim = 512

        norm = AdaLayerNormZero(embedding_dim, norm_type="fp32_layer_norm")

        assert norm.norm is not None

    def test_forward(self):
        """Test forward pass with direct embedding input."""
        batch_size, seq_len, embedding_dim = 4, 16, 512

        norm = AdaLayerNormZero(embedding_dim)
        x = randn(batch_size, seq_len, embedding_dim)
        emb = randn(batch_size, embedding_dim)

        x_out, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(x, emb=emb)

        assert x_out is not None
        assert gate_msa is not None
        assert shift_mlp is not None
        assert scale_mlp is not None
        assert gate_mlp is not None

    def test_forward__timestep_and_labels(self):
        """Test forward pass with timestep and class labels."""
        batch_size, seq_len, embedding_dim = 4, 16, 512
        num_embeddings = 10

        norm = AdaLayerNormZero(embedding_dim, num_embeddings)
        norm.eval()  # Disable dropout

        x = randn(batch_size, seq_len, embedding_dim)
        timestep = randn(batch_size)
        class_labels = randint(0, num_embeddings, (batch_size,))

        x_out, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(
            x, timestep=timestep, class_labels=class_labels
        )

        assert x_out is not None
        assert gate_msa is not None
        assert shift_mlp is not None
        assert scale_mlp is not None
        assert gate_mlp is not None
