# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from components.adapters.mapper import AdapterMapper


class BFLFluxAdapterMapper(AdapterMapper):
    """
    Maps BFL-format Flux LoRA adapter keys to BFL checkpoint weight keys.

    Handles both:
    - Fused adapters: double_blocks_0_img_attn_qkv (direct mapping)
    - Unfused adapters: double_blocks_0_img_attn_to_q (sliced mapping to qkv)

    For unfused adapters applied to fused checkpoints, returns slice info:
      ("double_blocks.0.img_attn.qkv.weight", (0, 0, 3072)) for to_q
      ("double_blocks.0.img_attn.qkv.weight", (0, 3072, 3072)) for to_k
      ("double_blocks.0.img_attn.qkv.weight", (0, 6144, 3072)) for to_v
    """

    def __init__(self, target_format: str = "bfl-fused", hidden_size: int = 3072):
        """
        Args:
            target_format: Format of the target model checkpoint
                - "bfl-fused": BFL format with fused QKV weights
                - "bfl-unfused": BFL format with separate Q/K/V weights
            hidden_size: Hidden dimension size (3072 for Flux dev)
        """
        self.target_format = target_format
        self.hidden_size = hidden_size

    def map_key(self, adapter_key: str) -> str | tuple | None:
        # All keys start with "lora_unet_"
        if not adapter_key.startswith("lora_unet_"):
            return None

        # Strip prefix
        key_body = adapter_key[len("lora_unet_"):]

        # Split into segments
        segments = key_body.split("_")

        if len(segments) < 3:
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

        # Handle double_blocks
        if block_type == "double_blocks":
            # Check for unfused Q/K/V patterns
            if "attn" in adapter_key:
                # Determine if img or txt attention
                is_img = "img" in adapter_key
                attn_prefix = "img_attn" if is_img else "txt_attn"

                if "to_q" in adapter_key or "to_k" in adapter_key or "to_v" in adapter_key:
                    # Unfused adapter -> needs slicing for fused checkpoint
                    if self.target_format == "bfl-fused":
                        qkv_key = f"double_blocks.{block_idx}.{attn_prefix}.qkv.weight"

                        # Determine which slice (Q/K/V)
                        if "to_q" in adapter_key:
                            return (qkv_key, (0, 0, self.hidden_size))
                        elif "to_k" in adapter_key:
                            return (qkv_key, (0, self.hidden_size, self.hidden_size))
                        elif "to_v" in adapter_key:
                            return (qkv_key, (0, self.hidden_size * 2, self.hidden_size))
                    else:
                        # Target is unfused, direct mapping
                        if "to_q" in adapter_key:
                            return f"double_blocks.{block_idx}.{attn_prefix}.to_q.weight"
                        elif "to_k" in adapter_key:
                            return f"double_blocks.{block_idx}.{attn_prefix}.to_k.weight"
                        elif "to_v" in adapter_key:
                            return f"double_blocks.{block_idx}.{attn_prefix}.to_v.weight"

                elif "attn_qkv" in adapter_key:
                    # Fused adapter
                    return f"double_blocks.{block_idx}.{attn_prefix}.qkv.weight"

                elif "attn_proj" in adapter_key or "proj" in adapter_key:
                    # Output projection
                    return f"double_blocks.{block_idx}.{attn_prefix}.proj.weight"

            # MLP layers
            elif "mlp" in adapter_key:
                is_img = "img" in adapter_key
                mlp_prefix = "img_mlp" if is_img else "txt_mlp"
                mlp_idx = subpath[-1]  # last segment is index
                return f"double_blocks.{block_idx}.{mlp_prefix}.{mlp_idx}.weight"

            # Modulation layers
            elif "mod" in adapter_key and "lin" in adapter_key:
                is_img = "img" in adapter_key
                mod_prefix = "img_mod" if is_img else "txt_mod"
                return f"double_blocks.{block_idx}.{mod_prefix}.lin.weight"

        # Handle single_blocks
        elif block_type == "single_blocks":
            # Single blocks use fused linear1 (Q/K/V/MLP all in one)
            if "linear1" in adapter_key:
                # Check if adapter has unfused keys
                if "to_q" in adapter_key:
                    if self.target_format == "bfl-fused":
                        return (f"single_blocks.{block_idx}.linear1.weight", (0, 0, self.hidden_size))
                    else:
                        return f"single_blocks.{block_idx}.to_q.weight"
                elif "to_k" in adapter_key:
                    if self.target_format == "bfl-fused":
                        return (f"single_blocks.{block_idx}.linear1.weight", (0, self.hidden_size, self.hidden_size))
                    else:
                        return f"single_blocks.{block_idx}.to_k.weight"
                elif "to_v" in adapter_key:
                    if self.target_format == "bfl-fused":
                        return (f"single_blocks.{block_idx}.linear1.weight", (0, self.hidden_size * 2, self.hidden_size))
                    else:
                        return f"single_blocks.{block_idx}.to_v.weight"
                else:
                    # Fused linear1
                    return f"single_blocks.{block_idx}.linear1.weight"

            elif "linear2" in adapter_key:
                return f"single_blocks.{block_idx}.linear2.weight"

            elif "modulation" in adapter_key or "mod" in adapter_key:
                return f"single_blocks.{block_idx}.modulation.lin.weight"

        return None
