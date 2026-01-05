"""
Tests for GELU activation module.
"""
from torch import randn

from components.tranformers.modules.activators.gelu import GELU


class TestGELU:
    """Test suite for GELU activation function."""

    def test_forward(self):
        batch_size, seq_len, dim_in, dim_out = 2, 16, 128, 256

        module = GELU(dim_in, dim_out)
        tensor = randn(batch_size, seq_len, dim_in)

        output = module(tensor)

        assert output is not None and output.numel() > 0
