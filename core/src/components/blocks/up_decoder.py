# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional

import torch
import torch.nn as nn

from .resnet import ResnetBlock
from .upsample import Upsample2D


class UpDecoderBlock2D(nn.Module):
    """
    Upsampling decoder block with residual connections.

    Args:
        input_channel_count: Number of input channels
        output_channel_count: Number of output channels
        layer_count: Number of ResNet layers in the block
        dropout: Dropout probability
        add_upsample: Whether to add upsampling at the end of the block
        group_count: Number of groups for GroupNorm in ResNet blocks
        epsilon: Epsilon for GroupNorm in ResNet blocks
        activation_function: Activation function for ResNet blocks
        output_scale_factor: Output scale factor for ResNet blocks
    """

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        layer_count: int = 1,
        dropout: float = 0.0,
        add_upsample: bool = True,
        group_count: int = 32,
        epsilon: float = 1e-6,
        activation_function: str = "silu",
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count

        # Build ResNet layers
        resnets = []
        for i in range(layer_count):
            input_channels = input_channel_count if i == 0 else output_channel_count

            resnets.append(
                ResnetBlock(
                    input_channel_count=input_channels,
                    output_channel_count=output_channel_count,
                    dropout=dropout,
                    group_count=group_count,
                    epsilon=epsilon,
                    non_linearity=activation_function,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # Optional upsampler
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(
                    channel_count=output_channel_count,
                    output_channel_count=output_channel_count,
                    use_conv=True,
                )
            ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling decoder block.

        Args:
            hidden_states: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height*2, width*2) if upsampling,
            otherwise (batch, out_channels, height, width)
        """
        # Process through ResNet blocks
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        # Optional upsampling
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
