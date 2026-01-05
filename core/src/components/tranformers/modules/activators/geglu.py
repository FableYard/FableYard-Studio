# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor
from torch.nn import Module, Linear, functional

# Check if NPU is available (torch_npu might not be installed)
try:
    from torch import is_torch_npu_available
    _npu_available = is_torch_npu_available()
except (ImportError, AttributeError):
    _npu_available = False


class GEGLU(Module):
    r"""
    A [variant](https://huggingface.co/papers/2002.05202) of the gated linear unit activation function.

    Parameters:
        input_channel_count (`int`): The number of channels in the input.
        output_channel_count (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, input_channel_count: int, output_channel_count: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(input_channel_count, output_channel_count * 2, bias=bias)

    def gelu(self, gate: Tensor) -> Tensor:
        return functional.gelu(gate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        if _npu_available:
            import torch_npu
            # using torch_npu.npu_geglu can run faster and save memory on NPU.
            return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        else:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)
