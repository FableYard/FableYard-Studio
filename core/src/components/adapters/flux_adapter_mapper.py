from core.src.components.adapters.mapper import AdapterMapper


class FluxAdapterMapper(AdapterMapper):
    """
    Maps Flux LoRA adapter keys to Flux transformer weight keys.
    Supports:
      - double_blocks fused attention (QKV + proj)
      - double_blocks MLP
      - double_blocks modulation lines
      - single_blocks linear / MLP / modulation
    """

    def map_key(self, adapter_key: str) -> str | None:
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

        block_type = segments[0]          # double_blocks or single_blocks
        block_idx = segments[1]           # numeric index
        subpath = segments[2:]            # rest

        # Handle double_blocks
        if block_type == "double_blocks":
            if "attn_qkv" in adapter_key:
                # fused attention QKV
                transformer_key = f"transformer.double_blocks.{block_idx}.attn.qkv.weight"
            elif "attn_proj" in adapter_key:
                transformer_key = f"transformer.double_blocks.{block_idx}.attn.proj.weight"
            elif "mlp" in adapter_key:
                # MLP layers: *_mlp_0 or *_mlp_2
                mlp_idx = subpath[-1]  # last segment is index
                transformer_key = f"transformer.double_blocks.{block_idx}.mlp.{mlp_idx}.weight"
            elif "mod_lin" in adapter_key:
                transformer_key = f"transformer.double_blocks.{block_idx}.modulation_lin.weight"
            else:
                # Unknown double_blocks pattern
                return None

            return transformer_key

        # Handle single_blocks
        elif block_type == "single_blocks":
            if "linear1" in adapter_key:
                transformer_key = f"transformer.single_blocks.{block_idx}.linear1.weight"
            elif "linear2" in adapter_key:
                transformer_key = f"transformer.single_blocks.{block_idx}.linear2.weight"
            elif "modulation_lin" in adapter_key or "mod_lin" in adapter_key:
                transformer_key = f"transformer.single_blocks.{block_idx}.modulation_lin.weight"
            else:
                return None

            return transformer_key

        # Unknown block type
        return None
