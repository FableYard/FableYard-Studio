from pathlib import Path

import torch

from core.src.components.adapters.adapter_processor import AdapterProcessor
from core.src.components.adapters.mapper import AdapterMapper
from core.src.components.adapters.utils import load_state_dict


class AdapterPatcher:
    """Patch transformer state_dict with adapters."""

    def __init__(self, transformer_state_dict: dict, model_type: str = 'flux'):
        self.state_dict = transformer_state_dict
        self.model_type = model_type
        self.pending_adapters = []  # [(AdapterProcessor, strength)]

    def add_adapter(self, path: Path, strength: float):
        state_dict = load_state_dict(path)
        # Create mapper with auto-detection based on adapter state_dict
        mapper = AdapterMapper.from_model_type(self.model_type, state_dict)
        print(f"  Mapper type: {type(mapper).__name__}")
        if hasattr(mapper, 'target_format'):
            print(f"  Target format: {mapper.target_format}")
        processor = AdapterProcessor(state_dict, mapper)
        print(f"  Created {len(processor.patches)} patches")
        print(f"  Mapped to {len(processor.mapped_patches)} model keys")
        if len(processor.patches) > 0 and len(processor.mapped_patches) == 0:
            test_key = list(processor.patches.keys())[0]
            result = mapper.map_key(test_key)
            print(f"  DEBUG: Test key '{test_key}' -> '{result}'")
        self.pending_adapters.append((processor, strength))

    def apply_patches(self):
        """
        Apply all staged adapters layer-by-layer.
        """
        patched_count = 0
        total_weights = 0

        for weight_key, weight in self.state_dict.items():
            if not torch.is_floating_point(weight):
                continue

            total_weights += 1
            delta_sum = None

            for processor, strength in self.pending_adapters:
                delta = processor.compute_delta(
                    transformer_key=weight_key,
                    strength=strength,
                    dtype=weight.dtype,
                    weight_shape=weight.shape
                )

                if delta is not None:
                    if delta.shape != weight.shape:
                        raise ValueError(f"Shape mismatch for {weight_key}: delta {delta.shape} vs weight {weight.shape}")
                    delta_sum = delta if delta_sum is None else delta_sum.add_(delta)

            if delta_sum is not None:
                weight.add_(delta_sum)
                patched_count += 1

        print(f"  Applied patches to {patched_count}/{total_weights} weights")

        # Release adapters
        self.pending_adapters.clear()
