# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

from safetensors.torch import load_file


def load_state_dict(path: Path, device: str = "cpu") -> dict:
    """Load adapter safetensors to specified device."""
    return load_file(path, device=device)
