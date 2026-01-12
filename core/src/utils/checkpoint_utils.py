"""Utilities for loading components from single checkpoint files."""
from pathlib import Path
from safetensors import safe_open
import torch


def is_checkpoint_file(path: Path | str) -> bool:
    """Check if path is a checkpoint file (.safetensors) or directory."""
    path = Path(path)
    return path.is_file() and path.suffix == '.safetensors'


def load_state_dict_from_checkpoint(
    checkpoint_path: Path | str,
    key_prefix: str,
    strip_prefix: bool = True,
    device: str = "cpu"
) -> dict:
    """
    Load only keys matching a prefix from a checkpoint file.

    Args:
        checkpoint_path: Path to .safetensors checkpoint
        key_prefix: Prefix to filter keys (e.g., "model.diffusion_model.")
        strip_prefix: If True, remove the prefix from returned keys
        device: Device to load tensors to

    Returns:
        State dict with matching keys
    """
    checkpoint_path = Path(checkpoint_path)
    state_dict = {}

    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for key in f.keys():
            if key.startswith(key_prefix):
                # Load tensor
                tensor = f.get_tensor(key)

                # Optionally strip prefix
                if strip_prefix:
                    new_key = key[len(key_prefix):]
                else:
                    new_key = key

                state_dict[new_key] = tensor

    return state_dict


def detect_checkpoint_format(checkpoint_path: Path | str) -> str:
    """
    Detect checkpoint format by examining keys.

    Args:
        checkpoint_path: Path to .safetensors checkpoint

    Returns:
        "bfl" - BFL format (single_blocks, double_blocks)
        "diffusers" - Diffusers format (transformer_blocks, single_transformer_blocks)
        "unknown" - Cannot determine
    """
    checkpoint_path = Path(checkpoint_path)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())[:50]  # Sample first 50 keys

    # Check for BFL format indicators
    has_single_blocks = any("single_blocks" in k for k in keys)
    has_double_blocks = any("double_blocks" in k for k in keys)

    # Check for diffusers format indicators
    has_transformer_blocks = any("transformer_blocks" in k for k in keys)
    has_single_transformer = any("single_transformer_blocks" in k for k in keys)

    if (has_single_blocks or has_double_blocks) and not (has_transformer_blocks or has_single_transformer):
        return "bfl"
    elif (has_transformer_blocks or has_single_transformer) and not (has_single_blocks or has_double_blocks):
        return "diffusers"
    else:
        return "unknown"
