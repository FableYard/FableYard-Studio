# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Tuple

import torch
from torch import LongTensor, Tensor, dtype
from torch.nn import Module, Linear, LayerNorm, SiLU

from components.tranformers.modules.embedders.timestep_label import CombinedTimestepLabelEmbeddings
from components.tranformers.modules.normalizers.layer_fp32 import FP32LayerNorm
from components.tranformers.modules.fused_ops import fused_adaptive_scale_shift


class AdaLayerNormZero(Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = SiLU().to(dtype=torch.bfloat16)
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias).to(dtype=torch.bfloat16)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: Tensor,
        timestep: Optional[Tensor] = None,
        class_labels: Optional[LongTensor] = None,
        hidden_dtype: Optional[dtype] = None,
        emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb.to(dtype=torch.bfloat16)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)

        # Fused normalization + scale + shift for memory bandwidth optimization
        normalized = self.norm(x)
        x = fused_adaptive_scale_shift(normalized, scale_msa, shift_msa)

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp