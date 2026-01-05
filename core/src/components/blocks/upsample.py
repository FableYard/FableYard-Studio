# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn as nn


class Upsample2D(nn.Module):
    """
    2D upsampling layer that doubles spatial dimensions.

    Args:
        channel_count: Number of input/output channels
        ouptut_channel_count: Number of output channels (defaults to channels if None)
        use_conv: If True, uses convolution for upsampling; otherwise uses transpose convolution
    """

    def __init__(
        self,
        channel_count: int,
        output_channel_count: int = None,
        use_conv: bool = True,
    ):
        super().__init__()

        # self.channels = channel_count
        # self.out_channels = output_channel_count or channel_count
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(
                channel_count,
                output_channel_count,
                kernel_size=3,
                padding=1
            )
        else:
            self.conv = nn.ConvTranspose2d(
                channel_count,
                output_channel_count,
                kernel_size=4,
                stride=2,
                padding=1
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Upsample the input tensor.

        Args:
            hidden_states: Input tensor of shape (batch, channels, height, width)

        Returns:
            Upsampled tensor of shape (batch, out_channels, height*2, width*2)
        """
        if self.use_conv:
            # Interpolate then convolve
            hidden_states = torch.nn.functional.interpolate(
                hidden_states,
                scale_factor=2.0,
                mode="nearest"
            )
            hidden_states = self.conv(hidden_states)
        else:
            # Use transpose convolution
            hidden_states = self.conv(hidden_states)

        return hidden_states
