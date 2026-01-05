# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Qwen3 Text Encoder Component
Handles Qwen3 text encoding for Z-Image pipeline using HuggingFace Transformers.
"""
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModel


class Qwen3TextEncoder:
    """
    Qwen3 text encoder using HuggingFace Transformers.
    Wraps Qwen3Model for text-to-image generation.
    """

    def __init__(self, component_path: str | Path, device: str | torch.device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_path = Path(component_path)

        # Load Qwen3Model from local files
        # Using AutoModel for flexibility with different Qwen3 variants
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False
        )

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Cache config attributes
        config = self.model.config
        self.hidden_size = getattr(config, 'hidden_size', None)
        self.num_hidden_layers = getattr(config, 'num_hidden_layers', None)
        self.num_attention_heads = getattr(config, 'num_attention_heads', None)
        self.vocab_size = getattr(config, 'vocab_size', None)

    @torch.no_grad()
    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Encode input tokens to embeddings.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
            attention_mask: Attention mask tensor (optional)
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return ModelOutput object
            **kwargs: Additional arguments passed to the model

        Returns:
            Hidden states tensor from the second-to-last layer
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Run model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # Return second-to-last hidden state (common practice for text-to-image models)
        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            return outputs.hidden_states[-2]
        else:
            return outputs.last_hidden_state

    @torch.no_grad()
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs
    ):
        """
        Direct call interface matching HuggingFace model signature.

        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor (optional)
            output_hidden_states: Whether to output all hidden states
            **kwargs: Additional model arguments

        Returns:
            Model outputs
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.model.set_input_embeddings(value)
