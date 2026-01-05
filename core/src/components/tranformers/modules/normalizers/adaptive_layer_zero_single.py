# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, SiLU

from components.tranformers.modules.fused_ops import fused_adaptive_scale_shift


class AdaLayerNormZeroSingle(Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        emb = self.linear(self.silu(emb.to(dtype=torch.bfloat16)))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)

        # Fused normalization + scale + shift for memory bandwidth optimization
        normalized = self.norm(x)
        x = fused_adaptive_scale_shift(normalized, scale_msa, shift_msa)

        return x, gate_msa