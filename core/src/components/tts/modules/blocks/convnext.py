# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ConvNeXt Block for Audio Processing

1D ConvNeXt block adapted for audio signal processing.
Based on the ConvNeXt architecture from Facebook Research.
"""

import torch
from torch import nn, Tensor


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block adapted for 1D audio signals.

    Architecture:
        - Depthwise convolution
        - Layer normalization
        - Two pointwise convolutions with GELU activation
        - Residual connection with optional layer scaling

    Args:
        dim: Number of input/output channels.
        intermediate_dim: Hidden dimension for the MLP.
        layer_scale_init_value: Initial value for layer scaling. None disables scaling.
        dw_kernel_size: Kernel size for depthwise convolution.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float = None,
        dw_kernel_size: int = 9,
    ):
        super().__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=dim
        )

        # Layer normalization (applied in channel-last format)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise convolutions (implemented as linear layers)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer scaling
        self.gamma = None
        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T) where B is batch size,
               C is channels (dim), and T is sequence length.

        Returns:
            Output tensor of same shape (B, C, T).
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Transpose to (B, T, C) for layer norm and linear layers
        x = x.transpose(1, 2)

        # MLP with layer norm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Apply layer scaling
        if self.gamma is not None:
            x = self.gamma * x

        # Transpose back to (B, C, T)
        x = x.transpose(1, 2)

        # Residual connection
        x = residual + x

        return x
