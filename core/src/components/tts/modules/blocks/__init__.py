# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Neural Network Blocks for TTS

Transformer blocks, convolutional blocks, etc.
"""

from .convnext import ConvNeXtBlock
from .istft import ISTFT

__all__ = ["ConvNeXtBlock", "ISTFT"]
