# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn as nn

from .attn_block import AttnBlock
from .resnet import ResnetBlock


class MidBlock(nn.Module):
    """
    Middle block for UNet-style/Transformer architecture with ResNet blocks.

    When attention is enabled, the block follows the pattern:
    ResNet0 -> Attention -> ResNet1 (-> ResNet2 -> ... if layer_count > 1)

    Args:
        input_channel_count: Number of input channels
        dropout: Dropout probability
        layer_count: Number of ResNet layers (default: 1)
        resnet_group_count: Number of groups for GroupNorm in ResNet blocks
        resnet_epsilon: Epsilon for GroupNorm in ResNet blocks
        resnet_activation_function: Activation function for ResNet blocks
        output_scale_factor: Output scale factor for ResNet blocks
        add_attention: Whether to add attention layers between ResNet blocks
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

        # When attention is enabled, we need at least 2 ResNet blocks
        # (one before attention, one after) to match diffusers weight structure
        actual_layer_count = max(layer_count + 1, 2) if add_attention else layer_count

        # Build ResNet blocks
        resnets = []
        for _ in range(actual_layer_count):
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

        # Build attention blocks
        if add_attention:
            self.attentions = nn.ModuleList([
                AttnBlock(
                    in_channels=input_channel_count,
                    num_groups=resnet_group_count,
                    eps=resnet_epsilon,
                )
            ])
        else:
            self.attentions = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the middle block.

        Args:
            hidden_states: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, in_channels, height, width)
        """
        # First ResNet block
        hidden_states = self.resnets[0](hidden_states)

        # Attention (if enabled)
        if self.attentions is not None:
            hidden_states = self.attentions[0](hidden_states)

        # Remaining ResNet blocks
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states)

        return hidden_states
