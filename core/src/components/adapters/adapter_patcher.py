from pathlib import Path
from typing import Union
import gc

import torch

from components.adapters.adapter_processor import AdapterProcessor
from components.adapters.mapper import AdapterMapper
from components.adapters.utils import load_state_dict

from utils import info


class AdapterPatcher:
    """Patch transformer state_dict with adapters (ComfyUI-style lazy evaluation)."""

    def __init__(self, model_type: str = 'flux'):
        self.model_type = model_type
        self.pending_adapters = []  # [(AdapterProcessor, strength)]
        self.state_dict = None

    def add_adapter(self, path: Union[str, Path], strength: float):
        """Load adapter and prepare for lazy application."""
        # Convert string path to Path object if needed
        if isinstance(path, str):
            path = Path(path)
        info(f"  Loading adapter: {path.name}")
        # Load to CPU to save GPU memory
        state_dict = load_state_dict(path, device="cpu")

        # Create mapper
        mapper = AdapterMapper.from_model_type(
            self.model_type,
            state_dict=state_dict,
            model_state_dict={}
        )
        info(f"    Mapper type: {type(mapper).__name__}")

        # Create processor (stores lora_up/lora_down, doesn't compute deltas yet)
        processor = AdapterProcessor(state_dict, mapper)
        info(f"    Mapped {len(processor.mapped_patches)} patches")

        self.pending_adapters.append((processor, strength))

        # Release loaded state dict
        del state_dict

    def apply_patches(self, state_dict: dict):
        """
        Apply adapters by computing deltas on-demand (ComfyUI style).
        Computes lora_up @ lora_down on GPU per weight, not all at once.
        """
        self.state_dict = state_dict
        info(f"Applying {len(self.pending_adapters)} adapters...")

        # Collect all keys that have patches
        keys_with_patches = set()
        for processor, _ in self.pending_adapters:
            keys_with_patches.update(processor.mapped_patches.keys())

        info(f"  Processing {len(keys_with_patches)} weights with patches")

        patched_count = 0
        weight_count = 0

        for weight_key in keys_with_patches:
            weight_count += 1
            if weight_count % 100 == 0:
                info(f"  Processed {weight_count}/{len(keys_with_patches)}, patched {patched_count}")

            weight = self.state_dict.get(weight_key)
            if weight is None:
                continue

            if not torch.is_floating_point(weight):
                continue

            # Check if weight is fp8
            is_fp8 = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            compute_dtype = torch.bfloat16 if is_fp8 else weight.dtype

            # Compute delta on-demand from each adapter
            delta_sum = None
            for processor, strength in self.pending_adapters:
                delta = processor.compute_delta(
                    transformer_key=weight_key,
                    strength=strength,
                    dtype=compute_dtype,
                    device=weight.device
                )

                if delta is not None:
                    if delta.shape != weight.shape:
                        raise ValueError(f"Shape mismatch for {weight_key}: delta {delta.shape} vs weight {weight.shape}")
                    delta_sum = delta if delta_sum is None else delta_sum.add_(delta)

            if delta_sum is not None:
                if is_fp8:
                    weight_compute = weight.to(dtype=torch.bfloat16)
                    weight_compute.add_(delta_sum)
                    self.state_dict[weight_key] = weight_compute
                else:
                    weight.add_(delta_sum)

                patched_count += 1

        info(f"  Applied {patched_count} patches")

        # Release adapters
        self.pending_adapters.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
