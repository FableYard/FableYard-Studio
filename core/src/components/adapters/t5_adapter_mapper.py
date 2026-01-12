# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from components.adapters.mapper import AdapterMapper


class T5AdapterMapper(AdapterMapper):
    """
    Maps T5 text encoder LoRA keys to HuggingFace T5 model weight keys.

    Handles keys like:
      - lora_te2_encoder_block_{N}_layer_0_SelfAttention_q
      - lora_te2_encoder_block_{N}_layer_0_SelfAttention_k
      - lora_te2_encoder_block_{N}_layer_0_SelfAttention_v
      - lora_te2_encoder_block_{N}_layer_0_SelfAttention_o
      - lora_te2_encoder_block_{N}_layer_1_DenseReluDense_wi
      - lora_te2_encoder_block_{N}_layer_1_DenseReluDense_wo

    Maps to HuggingFace T5 format:
      - encoder.block.{N}.layer.0.SelfAttention.q.weight
      - encoder.block.{N}.layer.1.DenseReluDense.wi.weight
    """

    def map_key(self, adapter_key: str) -> str | None:
        """
        Map T5 adapter key to model weight key.

        Returns:
            - str: Model weight key
            - None: Not a T5 key
        """
        # Only handle lora_te2 keys (T5)
        if not adapter_key.startswith("lora_te2_"):
            return None

        # Remove lora_te2_ prefix
        key_body = adapter_key[len("lora_te2_"):]

        # Replace underscores with dots, but preserve component names
        # encoder_block_0_layer_0_SelfAttention_q -> encoder.block.0.layer.0.SelfAttention.q

        # Parse structure
        if not key_body.startswith("encoder_block_"):
            return None

        remainder = key_body[len("encoder_block_"):]
        parts = remainder.split("_")

        if len(parts) < 5:
            return None

        # Extract block number, layer number, and component
        block_idx = parts[0]
        # parts[1] should be "layer"
        layer_idx = parts[2]
        # Remaining parts form the component path
        component_parts = parts[3:]

        # Reconstruct component path with appropriate dots
        component = ".".join(component_parts)

        return f"encoder.block.{block_idx}.layer.{layer_idx}.{component}.weight"
