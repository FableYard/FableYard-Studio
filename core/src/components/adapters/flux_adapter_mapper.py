# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from components.adapters.mapper import AdapterMapper


class FluxAdapterMapper(AdapterMapper):
    """
    Maps Flux LoRA adapter keys to Flux transformer weight keys.
    Supports:
      - double_blocks fused attention (QKV + proj)
      - double_blocks MLP
      - double_blocks modulation lines
      - single_blocks linear / MLP / modulation
    """

    def map_key(self, adapter_key: str) -> str | dict | None:
        # All keys start with "lora_unet_"
        if not adapter_key.startswith("lora_unet_"):
            return None

        # Strip prefix
        key_body = adapter_key[len("lora_unet_") :]

        # Split into segments
        # Format examples:
        # double_blocks_0_img_attn_qkv
        # double_blocks_0_img_mlp_0
        # double_blocks_0_img_mod_lin
        # single_blocks_0_linear1
        segments = key_body.split("_")

        if len(segments) < 3:
            # Unexpected / unknown pattern
            return None

        # Check if first two segments form "double_blocks" or "single_blocks"
        if segments[0] == "double" and segments[1] == "blocks":
            block_type = "double_blocks"
            block_idx = segments[2]
            subpath = segments[3:]
        elif segments[0] == "single" and segments[1] == "blocks":
            block_type = "single_blocks"
            block_idx = segments[2]
            subpath = segments[3:]
        else:
            return None

        # Handle double_blocks (maps to transformer_blocks in diffusers format)
        if block_type == "double_blocks":
            if "attn_qkv" in adapter_key:
                # fused attention QKV -> unfuse to separate Q, K, V
                # Split indices: Q=[0:3072], K=[3072:6144], V=[6144:9216]
                if "img" in adapter_key:
                    # img attention uses to_q, to_k, to_v
                    return {
                        "type": "unfused_multi",
                        "components": [
                            {"key": f"transformer_blocks.{block_idx}.attn.to_q.weight", "slice": [0, 3072]},
                            {"key": f"transformer_blocks.{block_idx}.attn.to_k.weight", "slice": [3072, 6144]},
                            {"key": f"transformer_blocks.{block_idx}.attn.to_v.weight", "slice": [6144, 9216]}
                        ]
                    }
                else:
                    # txt attention uses add_q_proj, add_k_proj, add_v_proj
                    return {
                        "type": "unfused_multi",
                        "components": [
                            {"key": f"transformer_blocks.{block_idx}.attn.add_q_proj.weight", "slice": [0, 3072]},
                            {"key": f"transformer_blocks.{block_idx}.attn.add_k_proj.weight", "slice": [3072, 6144]},
                            {"key": f"transformer_blocks.{block_idx}.attn.add_v_proj.weight", "slice": [6144, 9216]}
                        ]
                    }
            elif "attn" in adapter_key and ("to_q" in adapter_key or "to_k" in adapter_key or "to_v" in adapter_key):
                # Unfused BFL adapter keys (e.g., double_blocks_0_img_attn_to_q)
                # Map to diffusers unfused keys
                is_img = "img" in adapter_key

                if "to_q" in adapter_key:
                    component = "to_q"
                elif "to_k" in adapter_key:
                    component = "to_k"
                elif "to_v" in adapter_key:
                    component = "to_v"
                else:
                    return None

                if is_img:
                    transformer_key = f"transformer_blocks.{block_idx}.attn.{component}.weight"
                else:
                    # txt attention uses add_q_proj, add_k_proj, add_v_proj
                    if component == "to_q":
                        transformer_key = f"transformer_blocks.{block_idx}.attn.add_q_proj.weight"
                    elif component == "to_k":
                        transformer_key = f"transformer_blocks.{block_idx}.attn.add_k_proj.weight"
                    elif component == "to_v":
                        transformer_key = f"transformer_blocks.{block_idx}.attn.add_v_proj.weight"
                    else:
                        return None

                return transformer_key
            elif "attn_proj" in adapter_key:
                # Handle img vs txt attention
                if "img" in adapter_key:
                    transformer_key = f"transformer_blocks.{block_idx}.attn.to_out.0.weight"
                else:
                    transformer_key = f"transformer_blocks.{block_idx}.attn.to_add_out.weight"
            elif "mlp" in adapter_key:
                # MLP layers: *_mlp_0 or *_mlp_2
                mlp_idx = subpath[-1]  # last segment is index
                # img_mlp -> ff.net, txt_mlp -> ff_context.net
                if "img" in adapter_key:
                    if mlp_idx == "0":
                        transformer_key = f"transformer_blocks.{block_idx}.ff.net.0.proj.weight"
                    elif mlp_idx == "2":
                        transformer_key = f"transformer_blocks.{block_idx}.ff.net.2.weight"
                    else:
                        return None
                elif "txt" in adapter_key:
                    if mlp_idx == "0":
                        transformer_key = f"transformer_blocks.{block_idx}.ff_context.net.0.proj.weight"
                    elif mlp_idx == "2":
                        transformer_key = f"transformer_blocks.{block_idx}.ff_context.net.2.weight"
                    else:
                        return None
                else:
                    return None
            elif "mod_lin" in adapter_key:
                # img_mod -> norm1, txt_mod -> norm1_context
                if "img" in adapter_key:
                    transformer_key = f"transformer_blocks.{block_idx}.norm1.linear.weight"
                elif "txt" in adapter_key:
                    transformer_key = f"transformer_blocks.{block_idx}.norm1_context.linear.weight"
                else:
                    return None
            else:
                # Unknown double_blocks pattern
                return None

            return transformer_key

        # Handle single_blocks (maps to single_transformer_blocks in diffusers format)
        elif block_type == "single_blocks":
            # Check for unfused Q/K/V keys first
            if "to_q" in adapter_key:
                transformer_key = f"single_transformer_blocks.{block_idx}.attn.to_q.weight"
            elif "to_k" in adapter_key:
                transformer_key = f"single_transformer_blocks.{block_idx}.attn.to_k.weight"
            elif "to_v" in adapter_key:
                transformer_key = f"single_transformer_blocks.{block_idx}.attn.to_v.weight"
            elif "linear1" in adapter_key:
                # Fused linear1 (Q + K + V + MLP in BFL) -> unfuse to 4 components
                # Split indices: Q=[0:3072], K=[3072:6144], V=[6144:9216], MLP=[9216:21504]
                return {
                    "type": "unfused_multi",
                    "components": [
                        {"key": f"single_transformer_blocks.{block_idx}.attn.to_q.weight", "slice": [0, 3072]},
                        {"key": f"single_transformer_blocks.{block_idx}.attn.to_k.weight", "slice": [3072, 6144]},
                        {"key": f"single_transformer_blocks.{block_idx}.attn.to_v.weight", "slice": [6144, 9216]},
                        {"key": f"single_transformer_blocks.{block_idx}.proj_mlp.weight", "slice": [9216, 21504]}
                    ]
                }
            elif "linear2" in adapter_key:
                transformer_key = f"single_transformer_blocks.{block_idx}.proj_out.weight"
            elif "modulation_lin" in adapter_key or "mod_lin" in adapter_key:
                transformer_key = f"single_transformer_blocks.{block_idx}.norm.linear.weight"
            else:
                return None

            return transformer_key

        # Unknown block type
        return None
# from typing import Optional
#
# from components.adapters.adapter_role import AdapterRole
# from components.adapters.constants import DOUBLE_STREAM_KEY_MAP, SINGLE_STREAM_KEY_MAP
#
#
# def resolve_flux_key(key: str) -> Optional[AdapterRole]:
#     """
#     Given a full Flux state_dict key, return its semantic adapter role.
#     Returns None if the key should not receive adapters.
#     """
#
#     if key.startswith("transformer_blocks."):
#         for fragment, role in DOUBLE_STREAM_KEY_MAP.items():
#             if f".{fragment}." in key or key.endswith(f".{fragment}.weight"):
#                 return role
#
#     if key.startswith("single_transformer_blocks."):
#         for fragment, role in SINGLE_STREAM_KEY_MAP.items():
#             if f".{fragment}." in key or key.endswith(f".{fragment}.weight"):
#                 return role
#
#     return None