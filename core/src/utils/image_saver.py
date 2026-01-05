# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Image saving utility.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
import numpy as np


class ImageSaver:
    """
    Handles conversion of VAE decoder outputs to saved image files.

    Args:
        output_dir: Directory where images will be saved (default: "outputs")
        create_dir: Whether to create the output directory if it doesn't exist
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs",
        create_dir: bool = True,
    ):
        self.output_dir = Path(output_dir)

        if create_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor from VAE decoder output to a PIL Image.

        Args:
            tensor: Tensor of shape (1, 3, H, W) or (3, H, W) with values in any range

        Returns:
            PIL Image in RGB format
        """
        # Remove batch dimension if present
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)

        # Ensure we're working on CPU
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Convert to float32 for processing
        tensor = tensor.float()

        # Clamp to reasonable range and normalize to [0, 1]
        # VAE outputs are typically in range [-1, 1] or similar
        tensor = torch.clamp(tensor, -1.0, 1.0)
        tensor = (tensor + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

        # Convert to numpy array (C, H, W) -> (H, W, C)
        image_np = tensor.permute(1, 2, 0).numpy()

        # Convert to uint8 [0, 255]
        image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)

        # Create PIL Image
        image = Image.fromarray(image_np, mode="RGB")

        return image

    def save(
        self,
        tensor: torch.Tensor,
        filename: str = "example.png",
        subdirectory: Optional[str] = None,
    ) -> Path:
        """
        Save a tensor as a PNG image file.

        Args:
            tensor: Tensor from VAE decoder output
            filename: Name of the output file (default: "example.png")
            subdirectory: Optional subdirectory within output_dir

        Returns:
            Path to the saved image file
        """
        # Determine save path
        if subdirectory:
            save_dir = self.output_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir

        save_path = save_dir / filename

        # Convert and save
        image = self.tensor_to_pil(tensor)
        image.save(save_path)

        # Convert to absolute path for reliable access later
        save_path = save_path.resolve()

        # Lazy import to avoid circular dependency
        from utils import info, debug
        info(f"Image saved: {save_path}")
        debug(f"Image size: {image.size}")

        return save_path

    def save_grid(
        self,
        tensors: list[torch.Tensor],
        filename: str = "grid.png",
        cols: int = 4,
    ) -> Path:
        """
        Save multiple tensors as a grid image.

        Args:
            tensors: List of tensors from VAE decoder
            filename: Name of the output file
            cols: Number of columns in the grid

        Returns:
            Path to the saved grid image
        """
        images = [self.tensor_to_pil(t) for t in tensors]

        # Calculate grid dimensions
        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        # Get image dimensions from first image
        img_width, img_height = images[0].size

        # Create grid canvas
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid = Image.new("RGB", (grid_width, grid_height))

        # Paste images into grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * img_width
            y = row * img_height
            grid.paste(img, (x, y))

        # Save grid
        save_path = self.output_dir / filename
        grid.save(save_path)

        # Lazy import to avoid circular dependency
        from utils import info, debug
        info(f"Grid image saved: {save_path} ({n_images} images, {rows}x{cols})")
        debug(f"Grid size: {grid.size}")

        return save_path
