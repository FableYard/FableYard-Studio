# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Audio Saver Component

Saves audio tensors to WAV files.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor


class AudioSaver:
    """
    Saves audio tensors to WAV files.

    Supports saving PyTorch tensors or numpy arrays as 16-bit PCM WAV files.
    """

    def __init__(self, output_dir: Union[str, Path] = "output", sample_rate: int = 32000):
        """
        Initialize the audio saver.

        Args:
            output_dir: Directory to save audio files to.
            sample_rate: Sample rate of the audio in Hz.
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, audio: Union[Tensor, np.ndarray], filename: str) -> Path:
        """
        Save audio to a WAV file.

        Args:
            audio: Audio waveform as tensor or numpy array. Shape: (samples,) or (channels, samples).
            filename: Output filename (with or without .wav extension).

        Returns:
            Path to the saved file.
        """
        # Ensure filename has .wav extension
        if not filename.endswith(".wav"):
            filename = f"{filename}.wav"

        output_path = self.output_dir / filename

        # Convert to numpy if tensor
        if isinstance(audio, Tensor):
            audio = audio.detach().cpu().numpy()

        # Ensure 1D array for mono audio
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            elif audio.shape[1] == 1:
                audio = audio.squeeze(1)

        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write WAV file using scipy
        from scipy.io import wavfile
        wavfile.write(str(output_path), self.sample_rate, audio_int16)

        return output_path

    def save_batch(self, audios: list, prefix: str = "audio") -> list:
        """
        Save multiple audio files.

        Args:
            audios: List of audio tensors/arrays.
            prefix: Filename prefix.

        Returns:
            List of paths to saved files.
        """
        paths = []
        for i, audio in enumerate(audios):
            path = self.save(audio, f"{prefix}_{i:04d}")
            paths.append(path)
        return paths
