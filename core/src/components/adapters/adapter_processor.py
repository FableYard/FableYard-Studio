# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import torch


class AdapterProcessor:
    """
    Holds adapter patches for a single adapter.

    patches:
        {
            adapter_key: {
                alpha,
                lora_down,
                lora_up,
                rank
            }
        }

    mapped_patches:
        {
            transformer_weight_key: {
                alpha,
                lora_down,
                lora_up,
                rank
            }
        }
    """

    def __init__(self, state_dict: dict, mapper):
        self.state_dict = state_dict
        self.mapper = mapper

        self.patches = self._create_patches()
        self.mapped_patches = self._map_patches()

        # Release state dict but keep patches
        del self.state_dict

    def _create_patches(self) -> dict:
        """
        Create patches from state dict.

        Handles multiple naming conventions:
        - Standard: base_key.lora_down.weight, base_key.lora_up.weight, base_key.alpha
        - Diffusers: base_key.lora_A.weight, base_key.lora_B.weight (no alpha)
        """
        patches = {}
        processed = set()

        # First pass: collect patches with explicit alpha keys
        for key in self.state_dict.keys():
            if key.endswith('.alpha'):
                base_key = key[:-6]  # remove '.alpha'

                down_key = f"{base_key}.lora_down.weight"
                up_key = f"{base_key}.lora_up.weight"

                if down_key in self.state_dict and up_key in self.state_dict:
                    down = self.state_dict[down_key]
                    up = self.state_dict[up_key]

                    patches[base_key] = {
                        'alpha': self.state_dict[key].item(),
                        'lora_down': down,
                        'lora_up': up,
                        'rank': down.shape[0]
                    }
                    processed.add(base_key)

        # Second pass: collect diffusers-style patches (lora_A/lora_B, no alpha)
        for key in self.state_dict.keys():
            if key.endswith('.lora_A.weight') or key.endswith('.lora_B.weight'):
                # Extract base key
                if key.endswith('.lora_A.weight'):
                    base_key = key[:-len('.lora_A.weight')]
                else:
                    base_key = key[:-len('.lora_B.weight')]

                if base_key in processed:
                    continue  # Already processed with explicit alpha

                a_key = f"{base_key}.lora_A.weight"
                b_key = f"{base_key}.lora_B.weight"

                if a_key in self.state_dict and b_key in self.state_dict:
                    down = self.state_dict[a_key]
                    up = self.state_dict[b_key]

                    # Use rank as alpha if no explicit alpha (standard LoRA convention)
                    rank = down.shape[0]

                    patches[base_key] = {
                        'alpha': float(rank),  # Default alpha = rank
                        'lora_down': down,
                        'lora_up': up,
                        'rank': rank
                    }
                    processed.add(base_key)

        return patches

    def _map_patches(self) -> dict:
        """
        Map adapter keys to transformer weight keys.
        Stores lora_up/lora_down for on-demand delta computation.

        Returns:
            mapped: {
                transformer_key: {
                    'alpha': float,
                    'rank': int,
                    'lora_up': tensor,
                    'lora_down': tensor
                }
            }
        """
        mapped = {}

        for adapter_key, patch in self.patches.items():
            mapping_result = self.mapper.map_key(adapter_key)

            if mapping_result is None:
                continue

            # Check mapping type
            if isinstance(mapping_result, dict) and mapping_result.get("type") == "unfused_multi":
                # Unfused multi-component mapping: one fused adapter -> multiple model keys
                # Need to slice the lora_up tensor for each component
                components = mapping_result["components"]

                for component in components:
                    target_key = component["key"]
                    slice_range = component["slice"]  # [start, end]

                    # Store sliced patch for on-demand computation
                    lora_up_sliced = patch['lora_up'][slice_range[0]:slice_range[1], :]

                    sliced_patch = {
                        'alpha': patch['alpha'],
                        'rank': patch['rank'],
                        'lora_up': lora_up_sliced,
                        'lora_down': patch['lora_down']
                    }

                    if target_key in mapped:
                        print(f"WARNING: Duplicate mapping to {target_key}")

                    mapped[target_key] = sliced_patch

            elif isinstance(mapping_result, tuple):
                # Legacy sliced mapping (for reverse case: multiple adapters -> one model key)
                # Not used in current unfusing implementation
                pass

            else:
                # Direct mapping: transformer_key (string)
                transformer_key = mapping_result
                mapped[transformer_key] = patch

        return mapped

    def compute_delta(
        self,
        transformer_key: str,
        strength: float,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor | None:
        """
        Compute delta on-demand for a transformer weight (ComfyUI style).

        Args:
            transformer_key: The weight key in the transformer
            strength: Adapter strength multiplier
            dtype: Target dtype for the delta
            device: Target device for the delta

        Returns:
            Delta tensor with the same shape as the target weight, or None
        """
        patch = self.mapped_patches.get(transformer_key)
        if patch is None:
            return None

        # Compute LoRA delta on-demand: scale * (lora_up @ lora_down)
        # Move to target device first for fast GPU computation
        lora_up = patch['lora_up'].to(device=device, dtype=dtype)
        lora_down = patch['lora_down'].to(device=device, dtype=dtype)

        scale = (patch['alpha'] / patch['rank']) * strength
        delta = scale * (lora_up @ lora_down)

        return delta
