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
        Map adapter keys to transformer weight keys once.

        Returns:
            mapped: {
                transformer_key: [
                    (patch, slice_info),  # slice_info is None for direct patches
                    ...
                ]
            }
        """
        mapped = {}

        for adapter_key, patch in self.patches.items():
            mapping_result = self.mapper.map_key(adapter_key)

            if mapping_result is None:
                continue

            # Check if this is a sliced mapping or direct mapping
            if isinstance(mapping_result, tuple):
                # Sliced mapping: (transformer_key, slice_info)
                transformer_key, slice_info = mapping_result
            else:
                # Direct mapping: transformer_key (string)
                transformer_key = mapping_result
                slice_info = None

            # Store patch with its slice info
            # Multiple patches can target the same weight (e.g., q, k, v -> qkv)
            if transformer_key not in mapped:
                mapped[transformer_key] = []

            mapped[transformer_key].append((patch, slice_info))

        return mapped

    def compute_delta(
        self,
        transformer_key: str,
        strength: float,
        dtype: torch.dtype,
        weight_shape: tuple = None
    ) -> torch.Tensor | None:
        """
        Compute delta for a transformer weight if this adapter affects it.

        Args:
            transformer_key: The weight key in the transformer
            strength: Adapter strength multiplier
            dtype: Target dtype for the delta
            weight_shape: Shape of the target weight (needed for sliced patches)

        Returns:
            Delta tensor with the same shape as the target weight, or None
        """
        patches = self.mapped_patches.get(transformer_key)
        if patches is None:
            return None

        delta_accumulator = None

        for patch, slice_info in patches:
            scale = (patch['alpha'] / patch['rank']) * strength
            patch_delta = scale * (patch['lora_up'] @ patch['lora_down'])
            patch_delta = patch_delta.to(dtype)

            if slice_info is None:
                # Direct patch - entire weight is affected
                delta_accumulator = patch_delta if delta_accumulator is None else delta_accumulator.add_(patch_delta)
            else:
                # Sliced patch - only part of the weight is affected
                # slice_info format: (dimension, start_offset, length)
                if weight_shape is None:
                    raise ValueError(f"weight_shape required for sliced patch on {transformer_key}")

                # Create full-sized delta if needed
                if delta_accumulator is None:
                    delta_accumulator = torch.zeros(weight_shape, dtype=dtype, device=patch_delta.device)

                dim, start, length = slice_info

                # Apply patch to the appropriate slice
                if dim == 0:
                    delta_accumulator[start:start+length, :].add_(patch_delta)
                elif dim == 1:
                    delta_accumulator[:, start:start+length].add_(patch_delta)
                else:
                    raise ValueError(f"Unsupported slice dimension: {dim}")

        return delta_accumulator
