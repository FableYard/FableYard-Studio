# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor
from torch.nn import functional, Module, Linear


class GELU(Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        input_channel_count (`int`): The number of channels in the input.
        output_channel_count (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
            self,
            input_channel_count: int,
            output_channel_count: int,
            approximate: str = "none",
            bias: bool = True
    ):
        super().__init__()
        self.proj = Linear(input_channel_count, output_channel_count, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: Tensor) -> Tensor:
        return functional.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states