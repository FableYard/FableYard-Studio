# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional

import torch
import torch.nn as nn

from components.tranformers.modules.activators.utils import get_activation


class ResnetBlock(nn.Module):
    """
    A 2D residual block with optional upsampling.

    Args:
        input_channel_count: Number of input channels
        output_channel_count: Number of output channels (defaults to in_channels if None)
        dropout: Dropout probability
        group_count: Number of groups for GroupNorm
        epsilon: Epsilon for GroupNorm
        non_linearity: Activation function name (default: "silu")
        output_scale_factor: Scale factor for output (default: 1.0)
        conv_shortcut: Use convolutional shortcut instead of linear projection
    """

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: Optional[int] = None,
        dropout: float = 0.0,
        group_count: int = 32,
        epsilon: float = 1e-6,
        non_linearity: str = "silu",
        output_scale_factor: float = 1.0,
        conv_shortcut: bool = False,
    ):
        super().__init__()

        output_channel_count = output_channel_count or input_channel_count
        # self.input_channel_count = input_channel_count
        # self.output_channel_count = output_channel_count
        self.output_scale_factor = output_scale_factor

        # First normalization and convolution
        self.norm1 = nn.GroupNorm(
            num_groups=group_count,
            num_channels=input_channel_count,
            eps=epsilon,
            affine=True
        )
        self.conv1 = nn.Conv2d(
            input_channel_count,
            output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Second normalization and convolution
        self.norm2 = nn.GroupNorm(
            num_groups=group_count,
            num_channels=output_channel_count,
            eps=epsilon,
            affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            output_channel_count,
            output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Activation function
        self.nonlinearity = get_activation(non_linearity)

        # Skip connection for channel adjustment
        self.conv_shortcut = None
        if input_channel_count != output_channel_count:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    input_channel_count,
                    output_channel_count,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.conv_shortcut = nn.Conv2d(
                    input_channel_count,
                    output_channel_count,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            hidden_states: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        residual = hidden_states

        # First convolution path
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Second convolution path
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # Skip connection
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        # Combine and scale
        hidden_states = (hidden_states + residual) / self.output_scale_factor

        return hidden_states
