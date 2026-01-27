# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Model Path Resolver

Resolves model paths from (pipeline_type, family, name) to filesystem paths.
Handles case normalization and path validation.
"""

from pathlib import Path
from typing import Dict

# Case mapping for model families
# Input is lowercase, filesystem uses lowercase (matching actual directory structure)
FAMILY_CASE_MAP: Dict[str, str] = {
    "flux": "flux",
    "z": "z",
    "qwen": "qwen",
    "llama": "llama",
    "mistral": "mistral",
    "gemma": "gemma",
    "soprano": "soprano",
}

# Valid pipeline types
VALID_PIPELINE_TYPES = {"txt2img", "txt2txt", "txt2aud"}


def resolve_model_path(
    pipeline_type: str,
    model_family: str,
    model_name: str
) -> Path:
    """
    Resolve model path from (pipeline_type, family, name) to filesystem path.

    Args:
        pipeline_type: Type of pipeline (e.g., "txt2img")
        model_family: Model family name (e.g., "flux", "pony", "stablediffusion", "z")
        model_name: Model name/version (e.g., "dev-1", "version", "3-5")

    Returns:
        Path: Resolved absolute path to the model directory

    Raises:
        ValueError: If pipeline_type is invalid or model_family is unknown
        FileNotFoundError: If the resolved path does not exist

    Examples:
        >>> resolve_model_path("txt2img", "flux", "dev-1")
        Path("../user/models/txt2img/Flux/dev-1")
    """
    # Validate pipeline type
    if pipeline_type not in VALID_PIPELINE_TYPES:
        raise ValueError(
            f"Invalid pipeline_type: '{pipeline_type}'. "
            f"Valid types: {', '.join(sorted(VALID_PIPELINE_TYPES))}"
        )

    # Normalize model_family to lowercase for lookup
    model_family_lower = model_family.lower()

    # Get filesystem case for family
    if model_family_lower not in FAMILY_CASE_MAP:
        raise ValueError(
            f"Unknown model_family: '{model_family}'. "
            f"Valid families: {', '.join(sorted(FAMILY_CASE_MAP.keys()))}"
        )

    family_fs_case = FAMILY_CASE_MAP[model_family_lower]

    # Construct path: user/models/{pipeline_type}/{ModelFamily}/{model_name}
    # In Docker: /user/models (absolute path from volume mount)
    # In local dev: ../user/models (relative path from core directory)
    import os
    if os.path.exists("/user/models"):
        # Docker environment
        base_path = Path("/user/models")
    else:
        # Local development environment
        base_path = Path("..") / "user" / "models"

    model_path = base_path / pipeline_type / family_fs_case / model_name

    # Validate that path exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Expected structure: user/models/{pipeline_type}/{family_fs_case}/{model_name}/\n"
            f"Please ensure the model is installed in the correct location."
        )

    return model_path
