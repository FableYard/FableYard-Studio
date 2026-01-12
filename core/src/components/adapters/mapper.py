class AdapterMapper:
    """
    Maps adapter keys → transformer weight keys.

    Return:
      - transformer weight key (str), or
      - tuple (transformer_key, (dim, offset, size)) for sliced mappings, or
      - None if the adapter key does not map to a model weight

    Sliced mappings are used when adapter has unfused Q/K/V but model has fused QKV.
    Example: adapter key "to_q" maps to ("qkv.weight", (0, 0, 3072))
    """

    @staticmethod
    def from_model_type(
            model_type: str,
            state_dict: dict = None,
            model_state_dict: dict = None
    ) -> 'AdapterMapper':
        """
        Create appropriate mapper for model type.

        Args:
            model_type: Type of model (e.g., 'flux', 'clip', 't5')
            state_dict: Optional adapter state dict for adapter format detection
            model_state_dict: Optional model state dict for model format detection

        Returns:
            Appropriate AdapterMapper instance
        """
        if model_type == 'clip':
            from components.adapters.clip_adapter_mapper import CLIPAdapterMapper
            return CLIPAdapterMapper()

        elif model_type == 't5':
            from components.adapters.t5_adapter_mapper import T5AdapterMapper
            return T5AdapterMapper()

        elif model_type == 'flux':
            # Detect model format from model state dict if provided
            model_format = None
            if model_state_dict is not None:
                model_format = AdapterMapper._detect_model_format(model_state_dict)

            # Auto-detect adapter format
            if state_dict is not None:
                adapter_format = AdapterMapper._detect_adapter_format(state_dict)

                if adapter_format == 'diffusers':
                    # Z-style adapters (diffusion_model.layers)
                    from components.adapters.diffusers_flux_adapter_mapper import DiffusersFluxAdapterMapper
                    # Target format based on detected model format
                    target = model_format if model_format else "z"
                    return DiffusersFluxAdapterMapper(target_format=target)

                elif adapter_format == 'standard':
                    # Standard lora_unet adapters (BFL format)
                    # Check if model is BFL format
                    if model_format in ["bfl", "bfl-fused", "bfl-unfused"]:
                        from components.adapters.bfl_flux_adapter_mapper import BFLFluxAdapterMapper
                        # Use BFL mapper with model format as target
                        target_format = model_format if model_format.startswith("bfl") else "bfl-fused"
                        return BFLFluxAdapterMapper(target_format=target_format, hidden_size=3072)
                    else:
                        # Model is diffusers format, use diffusers mapper
                        from components.adapters.flux_adapter_mapper import FluxAdapterMapper
                        return FluxAdapterMapper()

            # Default based on model format if known
            if model_format == "bfl" or model_format == "bfl-fused":
                from components.adapters.bfl_flux_adapter_mapper import BFLFluxAdapterMapper
                return BFLFluxAdapterMapper(target_format="bfl-fused", hidden_size=3072)
            else:
                # Default to diffusers mapper
                from components.adapters.diffusers_flux_adapter_mapper import DiffusersFluxAdapterMapper
                return DiffusersFluxAdapterMapper(target_format="diffusers")

        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _detect_adapter_format(state_dict: dict) -> str:
        """
        Detect adapter format from state dict keys.

        Returns:
            'diffusers': Keys start with 'diffusion_model.layers'
            'standard': Keys start with 'lora_unet_'
        """
        sample_keys = list(state_dict.keys())[:20]

        # Check for diffusers format
        if any(key.startswith('diffusion_model.layers.') for key in sample_keys):
            return 'diffusers'

        # Check for standard format
        if any(key.startswith('lora_unet_') for key in sample_keys):
            return 'standard'

        # Default to standard
        return 'standard'

    @staticmethod
    def _detect_model_format(state_dict: dict) -> str:
        """
        Detect model format from state dict keys.

        Returns:
            'bfl-fused': BFL format with fused QKV (double_blocks.*.qkv)
            'bfl-unfused': BFL format with unfused Q/K/V (double_blocks.*.to_q)
            'diffusers': Diffusers format (transformer_blocks, single_transformer_blocks)
        """
        sample_keys = list(state_dict.keys())[:100]

        # Check for diffusers format
        if any('transformer_blocks' in key or 'single_transformer_blocks' in key for key in sample_keys):
            return 'diffusers'

        # Check for BFL format
        has_double_blocks = any('double_blocks' in key or 'single_blocks' in key for key in sample_keys)
        if has_double_blocks:
            # Determine if fused or unfused
            has_qkv = any('.qkv.' in key for key in sample_keys)
            has_to_q = any('.to_q.' in key for key in sample_keys)

            if has_qkv:
                return 'bfl-fused'
            elif has_to_q:
                return 'bfl-unfused'
            else:
                # Default to fused for BFL
                return 'bfl-fused'

        # Default to diffusers
        return 'diffusers'

    def map_key(self, adapter_key: str) -> str | tuple | None:
        raise NotImplementedError
