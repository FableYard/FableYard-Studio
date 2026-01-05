# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import nn

from components.tranformers.z_image.modules.constants import ADALN_EMBED_DIM


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        """Apply adaptive layer norm and output projection."""
        scale = 1.0 + self.adaLN_modulation(c)
        scale = scale.unsqueeze(1)
        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x
