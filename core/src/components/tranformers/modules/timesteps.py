# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor
from torch.nn import Module

from components.tranformers.modules.embedders.get_timestep_embedding import get_timestep_embedding


class Timesteps(Module):
    def __init__(self, channel_count: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.channel_count = channel_count
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: Tensor) -> Tensor:
        timestep_embedding = get_timestep_embedding(
            timesteps,
            self.channel_count,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return timestep_embedding