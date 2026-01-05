# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tensor utility functions for transformer outputs.
"""

import torch


def unpatchify(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 16,
) -> torch.Tensor:
    # Validate input
    assert not torch.isnan(latents).any(), \
        f"Input latents to unpatchify contain NaN! Shape: {latents.shape}"
    assert not torch.isinf(latents).any(), \
        f"Input latents to unpatchify contain Inf! Shape: {latents.shape}"

    batch_size, num_patches, channels = latents.shape

    # Convert image dimensions to unpacking dimensions
    # This gives us the intermediate size after accounting for both VAE and Flux packing
    height = height // vae_scale_factor
    width = width // vae_scale_factor

    # Reshape to 2D grid: (batch, height, width, channels//4, 2, 2)
    # This separates the 2x2 packing in the channel dimension
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)

    # Permute to interleave spatial dimensions: (batch, channels//4, height, 2, width, 2)
    # This is the KEY difference from standard unpatchification
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    # Merge the 2x2 blocks into spatial dimensions: (batch, channels//4, height*2, width*2)
    latents = latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    # Validate output
    assert not torch.isnan(latents).any(), \
        f"Output latents from unpatchify contain NaN! Shape: {latents.shape}"
    assert not torch.isinf(latents).any(), \
        f"Output latents from unpatchify contain Inf! Shape: {latents.shape}"

    return latents
