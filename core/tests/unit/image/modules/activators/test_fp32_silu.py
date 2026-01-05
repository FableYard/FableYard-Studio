"""
Tests for FP32SiLU activation module.
"""
from torch import randn

from components.tranformers.modules.activators.fp32_silu import FP32SiLU


class TestFP32SiLU:
    """Test suite for FP32SiLU activation function."""

    def test_forward(self):
        batch_size, seq_len, dim = 2, 16, 128

        module = FP32SiLU()
        tensor = randn(batch_size, seq_len, dim)

        output = module(tensor)

        assert output is not None and output.numel() > 0
