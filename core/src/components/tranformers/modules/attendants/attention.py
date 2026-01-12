# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

from typing import Optional

from torch import Tensor
from torch.nn import Module as TorchModule, Linear, Dropout, ModuleList

from components.tranformers.modules.normalizers.root_mean_squared import RMSNorm


class Attention(TorchModule):
    def __init__(
        self,
        query_dimension: int,
        head_count: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: Optional[int] = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * head_count
        self.query_dim = query_dimension
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dimension
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else head_count
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, epsilon=eps, elementwise_affine=elementwise_affine)
        self.norm_k = RMSNorm(dim_head, epsilon=eps, elementwise_affine=elementwise_affine)
        self.to_q = Linear(query_dimension, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dimension, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dimension, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = ModuleList([])
            self.to_out.append(Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, epsilon=eps)
            self.norm_added_k = RMSNorm(dim_head, epsilon=eps)

            self.add_q_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = Linear(self.inner_dim, query_dimension, bias=out_bias)

        self.processor = processor
        self._processor_params = set(inspect.signature(processor.__call__).parameters.keys())

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Filter kwargs to only include parameters the processor accepts
        kwargs = {k: w for k, w in kwargs.items() if k in self._processor_params}

        # Handle parameter name mapping for z_image processors
        # If processor expects freqs_cis, don't pass image_rotary_emb positionally
        if 'freqs_cis' in self._processor_params:
            # Map image_rotary_emb to freqs_cis if freqs_cis not already provided
            if 'freqs_cis' not in kwargs and image_rotary_emb is not None:
                kwargs['freqs_cis'] = image_rotary_emb
            # Don't pass image_rotary_emb positionally to avoid conflict
            return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # For other processors (Flux), pass image_rotary_emb positionally
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)
