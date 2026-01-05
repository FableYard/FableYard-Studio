# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
from torch.nn import Module, SiLU, GELU, Linear

from components.tranformers.modules.activators.fp32_silu import FP32SiLU


class PixArtAlphaTextProjection(Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = Linear(in_features=in_features, out_features=hidden_size, bias=True).to(dtype=torch.bfloat16)
        if act_fn == "gelu_tanh":
            self.act_1 = GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = Linear(in_features=hidden_size, out_features=out_features, bias=True).to(dtype=torch.bfloat16)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states