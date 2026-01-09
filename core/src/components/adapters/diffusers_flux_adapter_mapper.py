from core.src.components.adapters.mapper import AdapterMapper


class DiffusersFluxAdapterMapper(AdapterMapper):
    """
    Maps diffusers-format Flux LoRA adapter keys to Flux transformer weight keys.

    Handles adapters with patterns like:
      - diffusion_model.layers.{N}.attention.to_{q,k,v}
      - diffusion_model.layers.{N}.attention.to_out
      - diffusion_model.layers.{N}.feed_forward.w{1,2,3}
      - diffusion_model.layers.{N}.adaLN_modulation

    These are typically from Z image family or other diffusers-based trainers.

    Strategy:
      Since Z adapters use unfused Q/K/V and the diffusers Flux checkpoints
      ALSO use unfused Q/K/V (not fused as in BFL format), we can simply
      strip the "diffusion_model." prefix and the keys should match directly:

      - diffusion_model.layers.0.attention.to_q → layers.0.attention.to_q (Z model)
      - diffusion_model.layers.0.attention.to_q → transformer_blocks.0.attn.to_q (diffusers Flux)

      For diffusers Flux format, we need additional translation:
        layers → transformer_blocks
        attention → attn
    """

    def __init__(self, target_format: str = "z"):
        """
        Args:
            target_format: Target model checkpoint format
                - "z": Map to layers.X.attention format (Z model)
                - "diffusers": Map to transformer_blocks.X.attn format (diffusers Flux)
        """
        self.target_format = target_format

    def map_key(self, adapter_key: str) -> str | None:
        """
        Map adapter key to transformer weight key.

        For Z model format:
          diffusion_model.layers.0.attention.to_q → layers.0.attention.to_q

        For diffusers Flux format:
          diffusion_model.layers.0.attention.to_q → transformer_blocks.0.attn.to_q
          diffusion_model.layers.0.feed_forward.w1 → transformer_blocks.0.ff.net.0.proj
          diffusion_model.layers.0.feed_forward.w2 → transformer_blocks.0.ff.net.2
          diffusion_model.layers.0.adaLN_modulation.0 → transformer_blocks.0.norm1.linear

        Returns:
            - str: Direct weight key mapping
            - None: Unmapped key
        """
        # All keys should start with "diffusion_model.layers."
        if not adapter_key.startswith("diffusion_model.layers."):
            return None

        if self.target_format == "z":
            # Z model: Just strip "diffusion_model." prefix
            # diffusion_model.layers.0.attention.to_q → layers.0.attention.to_q
            return adapter_key[len("diffusion_model."):]

        elif self.target_format == "diffusers":
            # Diffusers Flux: Translate naming conventions
            # diffusion_model.layers.N → transformer_blocks.N
            key_without_prefix = adapter_key[len("diffusion_model.layers."):]
            parts = key_without_prefix.split(".")

            if len(parts) < 2:
                return None

            # Parse layer index
            try:
                layer_idx = int(parts[0])
            except ValueError:
                return None

            rest = parts[1:]
            component = rest[0] if rest else None

            if component == "attention":
                # attention.to_q → attn.to_q
                # attention.to_out.0 → attn.to_out.0
                return f"transformer_blocks.{layer_idx}.attn." + ".".join(rest[1:])

            elif component == "feed_forward":
                # feed_forward.w1 → ff.net.0.proj
                # feed_forward.w2 → ff.net.2
                # feed_forward.w3 → ff_context.net.0.proj (or skip if not applicable)
                if len(rest) < 2:
                    return None

                ff_component = rest[1]
                if ff_component == "w1":
                    return f"transformer_blocks.{layer_idx}.ff.net.0.proj"
                elif ff_component == "w2":
                    return f"transformer_blocks.{layer_idx}.ff.net.2"
                elif ff_component == "w3":
                    # w3 is gating - may be part of w1 in some implementations
                    # For now, skip (return None) as standard diffusers Flux doesn't have separate w3
                    return None

            elif component == "adaLN_modulation":
                # adaLN_modulation.0 → norm1.linear
                return f"transformer_blocks.{layer_idx}.norm1.linear"

        return None
