# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch.nn import Module, Linear, SiLU


class SwiGLU(Module):
    r"""
    A [variant](https://huggingface.co/papers/2002.05202) of the gated linear unit activation function. It's similar to
    `GEGLU` but uses SiLU / Swish instead of GeLU.

    Parameters:
        input_channel_count (`int`): The number of channels in the input.
        output_channel_count (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, input_channel_count: int, output_channel_count: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(input_channel_count, output_channel_count * 2, bias=bias)
        self.activation = SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * self.activation(gate)
        return hidden_states