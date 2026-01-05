# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor, sigmoid
from torch.nn import Module, Linear

class ApproximateGELU(Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://huggingface.co/papers/1606.08415).

    Parameters:
        input_channel_count (`int`): The number of channels in the input.
        output_channel_count (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, input_channel_count: int, output_channel_count: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(input_channel_count, output_channel_count, bias=bias)

    def forward(self, tensor: Tensor) -> Tensor:
        scale = 1.702 # TODO: Investigate value origin, possibly extract as a constant
        tensor = self.proj(tensor)
        tensor = tensor * sigmoid(scale * tensor)
        return tensor