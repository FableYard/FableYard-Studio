# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Image Saver Component
Saves the decoded image to disk.
"""
from typing import Any, Dict
from pathlib import Path
from components.base_component import PipelineComponent
from utils import ImageSaver


class ImageSaverComponent(PipelineComponent):
    """
    Saves the final decoded image to disk.
    """

    @property
    def component_name(self) -> str:
        return "Image Saver"

    def load(self) -> None:
        """Initialize image saver"""
        output_dir = self.config.get("output_dir", "outputs")
        self._image_saver = ImageSaver(output_dir=output_dir)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save decoded image to disk.

        Expected inputs:
            - decoded_image: Tensor (pixel space image)
            - job_id: str (from config, used for filename)

        Returns:
            - result_path: str (path to saved image)
        """
        decoded_image = inputs["decoded_image"]
        job_id = self.config.get("job_id", "output")

        # Save image
        save_path = self._image_saver.save(decoded_image, filename=f"{job_id}.png")

        return {
            "result_path": str(save_path)
        }

    def cleanup(self) -> None:
        """No cleanup needed for image saver"""
        super().cleanup()

    def __enter__(self):
        """Override to skip emitting component_start event (no frontend card)"""
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Override to skip emitting component_complete event (no frontend card)"""
        self.cleanup()
        return False
