# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional

from torch.nn import Module, Linear

from components.tranformers.modules.activators.utils import get_activation


class TimestepEmbedding(Module):
    def __init__(
        self,
        input_channel_count: int,
        time_embed_dim: int,
        activation_function: str = "silu",
        output_channel_count: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = Linear(input_channel_count, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = Linear(cond_proj_dim, input_channel_count, bias=False)
        else:
            self.cond_proj = None

        if output_channel_count is not None:
            time_embed_dim_out = output_channel_count
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        self.act = get_activation(activation_function)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
            
        return sample