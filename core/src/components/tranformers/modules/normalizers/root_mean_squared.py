# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from numbers import Integral

from torch import Size, ones, float16, float32, bfloat16, zeros, rsqrt
from torch.nn import Module, Parameter

# Check if NPU is available (torch_npu might not be installed)
try:
    from torch import is_torch_npu_available
    _npu_available = is_torch_npu_available()
except (ImportError, AttributeError):
    _npu_available = False
    def is_torch_npu_available():
        return False


class RMSNorm(Module):
    r"""
    RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

    Args:
        dimension_count (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        epsilon (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(self, dimension_count: int, epsilon: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine

        if isinstance(dimension_count, Integral):
            dimension_count = (dimension_count,)

        self.dim = Size(dimension_count)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = Parameter(ones(dimension_count))
            if bias:
                self.bias = Parameter(zeros(dimension_count))

    def forward(self, hidden_states):
        if is_torch_npu_available():
            import torch_npu

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [float16, bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

            hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.epsilon)[0]

            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * rsqrt(variance + self.epsilon)

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [float16, bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                hidden_states = hidden_states * self.weight
                
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias
            else:
                hidden_states = hidden_states.to(input_dtype)

        return hidden_states

