from pathlib import Path

from safetensors.torch import load_file


def load_state_dict(path: Path) -> dict:
    """Load adapter safetensors."""
    return load_file(path)
