# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from components.adapters.mapper import AdapterMapper


class CLIPAdapterMapper(AdapterMapper):
    """
    Maps CLIP text encoder LoRA keys to HuggingFace CLIPTextModel weight keys.

    Handles keys like:
      - lora_te1_text_model_encoder_layers_{N}_mlp_fc1
      - lora_te1_text_model_encoder_layers_{N}_mlp_fc2
      - lora_te1_text_model_encoder_layers_{N}_self_attn_q_proj
      - lora_te1_text_model_encoder_layers_{N}_self_attn_k_proj
      - lora_te1_text_model_encoder_layers_{N}_self_attn_v_proj
      - lora_te1_text_model_encoder_layers_{N}_self_attn_out_proj

    Maps to HuggingFace format:
      - text_model.encoder.layers.{N}.mlp.fc1.weight
      - text_model.encoder.layers.{N}.self_attn.q_proj.weight
    """

    def map_key(self, adapter_key: str) -> str | None:
        """
        Map CLIP adapter key to model weight key.

        Returns:
            - str: Model weight key
            - None: Not a CLIP key
        """
        # Only handle lora_te1 keys (CLIP)
        if not adapter_key.startswith("lora_te1_"):
            return None

        # Remove lora_te1_ prefix
        # lora_te1_text_model_encoder_layers_0_mlp_fc1 -> text_model_encoder_layers_0_mlp_fc1
        key_body = adapter_key[len("lora_te1_"):]

        # Replace underscores with dots to reconstruct path
        # text_model_encoder_layers_0_mlp_fc1 -> text_model.encoder.layers.0.mlp.fc1

        # Parse the structure
        if not key_body.startswith("text_model_encoder_layers_"):
            return None

        # Extract layer number and component
        remainder = key_body[len("text_model_encoder_layers_"):]
        parts = remainder.split("_")

        if len(parts) < 2:
            return None

        layer_idx = parts[0]
        component_parts = parts[1:]

        # Reconstruct the path
        if len(component_parts) == 2:
            # mlp_fc1 or mlp_fc2
            component = ".".join(component_parts)
            return f"text_model.encoder.layers.{layer_idx}.{component}.weight"
        elif len(component_parts) == 3:
            # self_attn_q_proj, self_attn_k_proj, etc.
            component = ".".join([component_parts[0] + "_" + component_parts[1], component_parts[2]])
            return f"text_model.encoder.layers.{layer_idx}.{component}.weight"

        return None
