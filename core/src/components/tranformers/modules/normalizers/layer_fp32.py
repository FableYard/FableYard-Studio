# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor
from torch.nn import LayerNorm, functional


class FP32LayerNorm(LayerNorm):
    def forward(self, inputs: Tensor) -> Tensor:
        origin_dtype = inputs.dtype
        return functional.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)