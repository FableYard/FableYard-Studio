# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Text-to-Speech Components

Modular TTS system with codec-based audio generation.
"""

from .audio_saver import AudioSaver
from .text_processor import TextProcessor

__all__ = [
    "AudioSaver",
    "TextProcessor",
]
