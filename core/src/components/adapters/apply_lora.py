# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Dict

import torch


def apply_lora(model: torch.nn.Module, lora_state_dict: Dict[str, torch.Tensor], alpha: float = 1.0):
    """
    Apply a Flux-style LoRA adapter to a model, ComfyUI-style:
    - Patches all keys that exist
    - No optional skipping
    - Applies deltas: W += alpha * (B @ A)

    Args:
        model: Flux model (with transformer_blocks and single_transformer_blocks)
        lora_state_dict: state_dict loaded from the LoRA file
        alpha: LoRA strength multiplier
    """

    for key, delta in lora_state_dict.items():
        # Determine target module in the model
        # First, split key into path and weight suffix
        if not key.endswith("weight"):
            continue  # skip biases, etc. for this loader

        parts = key.split(".")
        module_path = ".".join(parts[:-1])  # everything before ".weight"

        # Resolve module in the model
        target_module = model
        found = True
        for p in parts[:-1]:
            if hasattr(target_module, p):
                target_module = getattr(target_module, p)
            else:
                found = False
                break
        if not found:
            print(f"[LoRA] Skipped missing module: {module_path}")
            continue

        # Ensure target module has weight
        if not hasattr(target_module, "weight"):
            print(f"[LoRA] Skipped module without weight: {module_path}")
            continue

        # Apply the delta (linear addition, ComfyUI style)
        # delta is assumed precomputed (B @ A) in the LoRA state_dict
        target_weight = target_module.weight.data
        if target_weight.shape != delta.shape:
            print(f"[LoRA] Shape mismatch {module_path}: target {target_weight.shape} vs delta {delta.shape}")
            continue

        # Apply
        target_module.weight.data += alpha * delta
        print(f"[LoRA] Patched {module_path}, shape {delta.shape}")