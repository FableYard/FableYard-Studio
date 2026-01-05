# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn as nn

from .resnet import ResnetBlock


class MidBlock(nn.Module):
    """
    Middle block for UNet-style/Transformer architecture with ResNet blocks.

    Args:
        input_channel_count: Number of input channels
        dropout: Dropout probability
        layer_count: Number of ResNet layers (default: 1)
        resnet_group_count: Number of groups for GroupNorm in ResNet blocks
        resnet_epsilon: Epsilon for GroupNorm in ResNet blocks
        resnet_activation_function: Activation function for ResNet blocks
        output_scale_factor: Output scale factor for ResNet blocks
        add_attention: Whether to add attention layers (currently not implemented, keeping simple)
    """

    def __init__(
        self,
        input_channel_count: int,
        dropout: float = 0.0,
        layer_count: int = 1,
        resnet_group_count: int = 32,
        resnet_epsilon: float = 1e-6,
        resnet_activation_function: str = "silu",
        output_scale_factor: float = 1.0,
        add_attention: bool = True,
    ):
        super().__init__()

        self.add_attention = add_attention

        # Build ResNet blocks
        resnets = []
        for _ in range(layer_count):
            resnets.append(
                ResnetBlock(
                    input_channel_count=input_channel_count,
                    output_channel_count=input_channel_count,
                    dropout=dropout,
                    group_count=resnet_group_count,
                    epsilon=resnet_epsilon,
                    non_linearity=resnet_activation_function,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # Note: Attention blocks could be added here if needed in the future
        # For now, keeping it simple with just ResNet blocks
        self.attentions = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the middle block.

        Args:
            hidden_states: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, in_channels, height, width)
        """
        # Process through ResNet blocks
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states
