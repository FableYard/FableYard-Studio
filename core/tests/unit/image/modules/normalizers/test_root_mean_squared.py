"""
Tests for RMSNorm module.
"""
from torch import all, float16, isfinite, randn

from components.tranformers.modules.normalizers.root_mean_squared import RMSNorm


class TestRMSNorm:
    """Test suite for RMSNorm normalization layer."""

    def test_initialization__basic(self):
        """Test basic initialization."""
        dim = 128
        eps = 1e-5

        norm = RMSNorm(dim, eps)

        assert norm.epsilon == eps
        assert norm.elementwise_affine is True
        assert norm.weight is not None
        assert norm.bias is None

    def test_forward__basic(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 4, 16, 128
        eps = 1e-5

        norm = RMSNorm(dim, eps)
        hidden_states = randn(batch_size, seq_len, dim)

        output = norm(hidden_states)

        assert output is not None
        assert output.numel() > 0

    def test_forward__with_bias(self):
        """Test forward pass with bias."""
        batch_size, seq_len, dim = 4, 16, 128
        eps = 1e-5

        norm = RMSNorm(dim, eps, elementwise_affine=True, bias=True)
        hidden_states = randn(batch_size, seq_len, dim)

        output = norm(hidden_states)

        assert output.shape == hidden_states.shape
        assert all(isfinite(output))

    def test_forward__float16(self):
        """Test forward pass with float16 dtype."""
        batch_size, seq_len, dim = 4, 16, 128
        eps = 1e-5

        norm = RMSNorm(dim, eps)
        # Convert weight to float16
        norm.weight.data = norm.weight.data.to(float16)

        hidden_states = randn(batch_size, seq_len, dim)

        output = norm(hidden_states)

        assert output.shape == hidden_states.shape
        assert all(isfinite(output))

    def test_forward__normalizes_correctly(self):
        """Test that RMS normalization actually normalizes."""
        batch_size, seq_len, dim = 2, 8, 64
        eps = 1e-5

        norm = RMSNorm(dim, eps, elementwise_affine=False)
        hidden_states = randn(batch_size, seq_len, dim) * 10  # Scale up input

        output = norm(hidden_states)

        # After RMS norm (without affine), RMS should be approximately 1
        rms = (output.pow(2).mean(-1, keepdim=True).sqrt())
        # RMS should be close to 1 for each position
        assert all((rms - 1).abs() < 0.1)
