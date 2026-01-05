# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Latent Space Generator for diffusion models.

Generates random latent tensors with noise using torch.Generator for reproducibility.
Handles packed latent format (2x2 packing scheme) and positional embeddings.
"""

from typing import Optional, Tuple, Union

import torch


class LatentGenerator:
    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        vae_downsampling_factor: int = 16,
        latent_channels: int = 16,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.height = height
        self.width = width
        self.vae_downsampling_factor = vae_downsampling_factor
        self.latent_channels = latent_channels
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        # Calculate unpacked latent dimensions (before 2x2 packing)
        # We generate at 2x the VAE's natural resolution to account for 2x2 packing
        vae_latent_h = height // vae_downsampling_factor
        vae_latent_w = width // vae_downsampling_factor
        self.unpacked_height = 2 * vae_latent_h
        self.unpacked_width = 2 * vae_latent_w

        # After packing, dimensions are halved
        self.packed_height = self.unpacked_height // 2
        self.packed_width = self.unpacked_width // 2
        self.sequence_length = self.packed_height * self.packed_width

        # Packed channels (2x2 packing = 4x channels)
        self.packed_channels = latent_channels * 4

    def generate(
        self,
        batch_size: int = 1,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random latents and positional IDs.

        Creates noise in the unpacked spatial format, applies Flux's 2x2 packing
        transformation, and generates corresponding positional embeddings (img_ids).

        Args:
            batch_size: Number of latent samples to generate (default: 1)
            seed: Optional random seed for reproducibility. If provided, creates
                a new generator with this seed. Ignored if generator is provided.
            generator: Optional torch.Generator for reproducible random number
                generation. If not provided and seed is given, creates one automatically.

        Returns:
            latents: Packed latent tensor of shape (batch_size, sequence_length, packed_channels)
                Example: (1, 1024, 64) for 512x512 image
            img_ids: Positional embeddings of shape (sequence_length, 3)
                Each row is [0, row_index, col_index] for 2D grid positions

        Example:
            >>> # Using a seed for reproducibility
            >>> latents, img_ids = generator.generate(batch_size=2, seed=42)

            >>> # Using a pre-configured torch.Generator
            >>> rng = torch.Generator(device="cuda").manual_seed(123)
            >>> latents, img_ids = generator.generate(batch_size=1, generator=rng)
        """
        # Create generator if seed is provided but generator is not
        if generator is None and seed is not None:
            # Try to create device-specific generator (not supported on older PyTorch)
            try:
                generator = torch.Generator(device=self.device)
            except TypeError:
                generator = torch.Generator()
            generator.manual_seed(seed)

        # Generate random noise in unpacked spatial format
        unpacked_latents = torch.randn(
            (batch_size, self.latent_channels, self.unpacked_height, self.unpacked_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Validate unpacked latents
        assert not torch.isnan(unpacked_latents).any(), \
            f"Generated unpacked_latents contain NaN! Shape: {unpacked_latents.shape}, dtype: {unpacked_latents.dtype}"
        assert not torch.isinf(unpacked_latents).any(), \
            f"Generated unpacked_latents contain Inf! Shape: {unpacked_latents.shape}, dtype: {unpacked_latents.dtype}"

        # Pack using 2x2 packing scheme
        packed_latents = self._pack_latents(unpacked_latents)

        # Validate packed latents
        assert not torch.isnan(packed_latents).any(), \
            f"Packed latents contain NaN! Shape: {packed_latents.shape}, dtype: {packed_latents.dtype}"
        assert not torch.isinf(packed_latents).any(), \
            f"Packed latents contain Inf! Shape: {packed_latents.shape}, dtype: {packed_latents.dtype}"

        # Generate positional embeddings
        img_ids = self.generate_img_ids(self.packed_height, self.packed_width)

        return packed_latents, img_ids

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack spatial latents using 2x2 packing scheme.

        Transforms unpacked spatial latents into packed sequence format by grouping
        2x2 spatial blocks into the channel dimension. This preserves spatial locality
        while creating a sequence representation.

        Args:
            latents: Unpacked spatial latents of shape (batch, channels, height, width)

        Returns:
            packed: Packed latents of shape (batch, sequence_length, channels*4)
                where sequence_length = (height//2) * (width//2)

        Example:
            >>> unpacked = torch.randn(1, 16, 64, 64)
            >>> packed = generator._pack_latents(unpacked)
            >>> packed.shape  # torch.Size([1, 1024, 64])
        """
        b, c, h, w = latents.shape

        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"Latent height and width must be even for 2x2 packing, got {h}x{w}"
            )

        # Split into 2x2 blocks: (B, C, H//2, 2, W//2, 2)
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)

        # Permute to bring spatial blocks into channel dimension
        # (B, C, H//2, 2, W//2, 2) -> (B, H//2, W//2, C, 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()

        # Flatten to sequence format: (B, H//2 * W//2, C * 4)
        packed = latents.view(b, (h // 2) * (w // 2), c * 4)

        return packed

    def generate_img_ids(
        self,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Generate img_ids for positional embeddings.

        Creates a 2D grid of positional indices used by rotary position
        embeddings. Each position in the flattened latent sequence gets assigned
        its (row, col) coordinates in the spatial grid.

        IMPORTANT: The order is [0, row, col], not [row, col, 0]. The first
        dimension is always 0 and reserved for future use in Flux architecture.

        Args:
            height: Height of the packed latent grid (e.g., 32 for 512x512 image)
            width: Width of the packed latent grid (e.g., 32 for 512x512 image)

        Returns:
            img_ids: Tensor of shape (height*width, 3) where each row is
                [0, row_index, col_index]. Always returned as float32 regardless
                of the generator's dtype setting.

        Example:
            >>> img_ids = generator.generate_img_ids(32, 32)
            >>> img_ids.shape  # torch.Size([1024, 3])
            >>> img_ids[0]     # tensor([0., 0., 0.]) - top-left, first dim is always 0
            >>> img_ids[1]     # tensor([0., 0., 1.]) - position (0, 1)
            >>> img_ids[32]    # tensor([0., 1., 0.]) - position (1, 0)
        """
        # Create 2D grid of positions (always float32 for positional embeddings)
        rows = torch.arange(height, device=self.device, dtype=torch.float32)
        cols = torch.arange(width, device=self.device, dtype=torch.float32)

        # Generate meshgrid for all positions
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")

        # Create zeros for first dimension
        zeros = torch.zeros((height, width), device=self.device, dtype=torch.float32)

        # Stack in correct order: [0, row, col]
        img_ids = torch.stack([zeros, grid_rows, grid_cols], dim=-1)

        # Flatten to (height*width, 3)
        img_ids = img_ids.reshape(height * width, 3)

        return img_ids

    def __repr__(self) -> str:
        """String representation of the LatentGenerator."""
        return (
            f"LatentGenerator(\n"
            f"  height={self.height}, width={self.width},\n"
            f"  unpacked_shape=({self.latent_channels}, {self.unpacked_height}, {self.unpacked_width}),\n"
            f"  packed_shape=({self.sequence_length}, {self.packed_channels}),\n"
            f"  vae_downsampling={self.vae_downsampling_factor}x,\n"
            f"  device={self.device}, dtype={self.dtype}\n"
            f")"
        )
