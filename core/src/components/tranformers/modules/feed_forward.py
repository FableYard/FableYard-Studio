# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional

from torch import Tensor
from  torch.nn import Module, Sequential, Dropout, Linear

from components.tranformers.modules.activators.approximate_gelu import ApproximateGELU
from components.tranformers.modules.activators.geglu import GEGLU
from components.tranformers.modules.activators.gelu import GELU
from components.tranformers.modules.activators.linear import LinearActivation
from components.tranformers.modules.activators.swiglu import SwiGLU


class FeedForward(Module):
    r"""
    A feed-forward layer.

    Parameters:
        input_channel_count (`int`): The number of channels in the input.
        output_channel_count (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        multiplier (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_function (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: Optional[int] = None,
        multiplier: int = 4,
        dropout: float = 0.0,
        activation_function: str = "geglu",
        final_dropout: bool = False,
        inner_dimension=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dimension is None:
            inner_dimension = int(input_channel_count * multiplier)

        output_channel_count = output_channel_count if output_channel_count is not None else input_channel_count
        act_fn = None

        if activation_function == "gelu":
            act_fn = GELU(input_channel_count, inner_dimension, bias=bias)
        elif activation_function == "gelu-approximate":
            act_fn = GELU(input_channel_count, inner_dimension, approximate="tanh", bias=bias)
        elif activation_function == "geglu":
            act_fn = GEGLU(input_channel_count, inner_dimension, bias=bias)
        elif activation_function == "geglu-approximate":
            act_fn = ApproximateGELU(input_channel_count, inner_dimension, bias=bias)
        elif activation_function == "swiglu":
            act_fn = SwiGLU(input_channel_count, inner_dimension, bias=bias)
        elif activation_function == "linear-silu":
            act_fn = LinearActivation(input_channel_count, inner_dimension, bias=bias, activation="silu")

        modules = [
            act_fn,
            Dropout(dropout),
            Linear(inner_dimension, output_channel_count, bias=bias)
        ]
        if final_dropout:
            modules.append(Dropout(dropout))
        self.net = Sequential(*modules)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states)