# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch import Tensor, float32
from torch.nn import Module, functional


class FP32SiLU(Module):
    r"""
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inputs: Tensor) -> Tensor:
        return functional.silu(inputs.float(), inplace=False).to(float32)
