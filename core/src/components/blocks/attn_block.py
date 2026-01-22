# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    """
    Spatial self-attention block for VAE with FP32 upcasting.

    This block applies self-attention over spatial dimensions (H*W) to capture
    long-range dependencies. FP32 upcasting is used for Q/K similarity computation
    to ensure numerical stability, which is critical for VAE image quality.

    Args:
        in_channels: Number of input/output channels
        num_groups: Number of groups for GroupNorm (default: 32)
        eps: Epsilon for GroupNorm (default: 1e-6)
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )

        # Q, K, V projections using Linear layers (matches diffusers weight structure)
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)

        # Output projection (wrapped in Sequential to match diffusers weight structure)
        self.to_out = nn.Sequential(
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention block.

        Args:
            hidden_states: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape

        # Normalize
        hidden_states = self.group_norm(hidden_states)

        # Reshape for linear layers: (B, C, H, W) -> (B, H*W, C)
        hidden_states = hidden_states.view(batch, channels, -1).transpose(1, 2)

        # Compute Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Add head dimension: (B, H*W, C) -> (B, 1, H*W, C)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # CRITICAL: FP32 upcast for numerical stability
        # This prevents quality degradation (squiggly line artifacts) in VAE decoding
        original_dtype = q.dtype
        hidden_states = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(original_dtype)

        # Remove head dimension: (B, 1, H*W, C) -> (B, H*W, C)
        hidden_states = hidden_states.squeeze(1)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        hidden_states = hidden_states.transpose(1, 2).view(batch, channels, height, width)

        # Residual connection
        return hidden_states + residual
