class AdapterMapper:
    """
    Maps adapter keys → transformer weight keys.

    Return:
      - transformer weight key (str), or
      - tuple (transformer_key, slice_info) for sliced mappings, or
      - None if the adapter key does not map to a model weight
    """

    @staticmethod
    def from_model_type(model_type: str, state_dict: dict = None) -> 'AdapterMapper':
        """
        Create appropriate mapper for model type.

        Args:
            model_type: Type of model (e.g., 'flux')
            state_dict: Optional adapter state dict for auto-detection

        Returns:
            Appropriate AdapterMapper instance
        """
        if model_type == 'flux':
            # Auto-detect adapter format
            if state_dict is not None:
                format_type = AdapterMapper._detect_adapter_format(state_dict)
                if format_type == 'diffusers':
                    # Diffusers format adapters - need to determine target model format
                    # Default to "z" format for now (can be made configurable)
                    from core.src.components.adapters.diffusers_flux_adapter_mapper import DiffusersFluxAdapterMapper
                    return DiffusersFluxAdapterMapper(target_format="z")
                elif format_type == 'standard':
                    from core.src.components.adapters.flux_adapter_mapper import FluxAdapterMapper
                    return FluxAdapterMapper()

            # Default to standard Flux mapper
            from core.src.components.adapters.flux_adapter_mapper import FluxAdapterMapper
            return FluxAdapterMapper()

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

    def map_key(self, adapter_key: str) -> str | tuple | None:
        raise NotImplementedError
