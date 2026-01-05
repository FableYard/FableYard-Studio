# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import List, Tuple

from torch import Tensor, float32, float64, cat
from torch.nn import Module

from components.tranformers.modules.embedders.get_1d_rotary_pos_embed import get_1d_rotary_pos_embed


class PositionalEmbedder(Module):
    def __init__(self, theta: int, axes_dimension: List[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dimension = axes_dimension

    def forward(self, ids: Tensor) -> Tuple[Tensor, Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = float32 if (is_mps or is_npu) else float64

        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dimension[i],
                pos[:, i],
                theta=self.theta,
                frequency_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin
