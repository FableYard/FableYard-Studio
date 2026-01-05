# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch.nn import Module, Linear

from components.tranformers.modules.activators.utils import get_activation


class LinearActivation(Module):
    def __init__(
            self,
            input_channel_count: int,
            output_channel_count: int,
            bias: bool = True,
            activation: str = "silu"
    ):
        super().__init__()
        self.proj = Linear(input_channel_count, output_channel_count, bias=bias)
        self.activation = get_activation(activation)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)