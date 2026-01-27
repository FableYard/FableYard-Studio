# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Base TTS Pipeline

Abstract base class for text-to-speech pipelines.
Provides common infrastructure for TTS model execution.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch


class BaseTTSPipeline(ABC):
    """
    Abstract base class for TTS pipelines.

    Provides common setup and utility methods for all TTS implementations.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        prompt: str,
        seed: int = -1,
        device: Optional[str] = None,
    ):
        """
        Initialize TTS pipeline.

        Args:
            model_path: Path to model directory.
            prompt: Text to synthesize.
            seed: Random seed (-1 for random).
            device: Device to use (None for auto-detect).
        """
        self.model_path = Path(model_path)
        self.prompt = prompt
        self.seed = seed

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _setup_seed(self) -> int:
        """
        Set up random seed.

        Returns:
            The seed that was used.
        """
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            return self.seed
        else:
            import random
            random_seed = random.randint(0, 2147483647)
            torch.manual_seed(random_seed)
            return random_seed

    def _get_output_path(self, suffix: str = "tts") -> Path:
        """
        Generate output file path.

        Args:
            suffix: Suffix to add to filename.

        Returns:
            Path to output file.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).parent.parent.parent.parent.parent
        output_dir = project_root / "user" / "output" / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{timestamp}_{suffix}.wav"

    @abstractmethod
    def execute(self) -> str:
        """
        Execute the TTS pipeline.

        Returns:
            Path to the generated audio file.
        """
        pass

    def cleanup(self):
        """Release resources."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
