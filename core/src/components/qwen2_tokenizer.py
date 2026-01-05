# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Qwen2 Tokenizer Component
Handles Qwen2 tokenization for Z-Image pipeline using HuggingFace Transformers.
"""
from pathlib import Path
from typing import List, Union, Optional

import torch
from transformers import AutoTokenizer


class Qwen2Tokenizer:
    """
    Qwen2 tokenizer using HuggingFace Transformers AutoTokenizer.
    Supports chat template application for Z-Image models.
    """

    def __init__(self, component_path: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_path = Path(component_path)

        # Load tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False
        )

        # Cache config attributes
        self.vocab_size = self.tokenizer.vocab_size
        self.model_max_length = self.tokenizer.model_max_length
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        **kwargs
    ) -> Union[str, List[int]]:
        """
        Apply chat template to messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenize: Whether to tokenize the output
            add_generation_prompt: Whether to add generation prompt
            enable_thinking: Whether to enable thinking mode (Qwen-specific)
            **kwargs: Additional arguments passed to apply_chat_template

        Returns:
            Formatted string or token IDs depending on tokenize parameter
        """
        # Build kwargs for apply_chat_template
        template_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs
        }

        # Add enable_thinking if supported
        if enable_thinking and hasattr(self.tokenizer, 'chat_template'):
            template_kwargs["enable_thinking"] = enable_thinking

        return self.tokenizer.apply_chat_template(
            messages,
            **template_kwargs
        )

    def encode(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> dict:
        """
        Encode text to token IDs.

        Args:
            text: Text or list of texts to encode
            padding: Padding strategy ('max_length', True, False)
            max_length: Maximum sequence length (uses model_max_length if None)
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        max_length = max_length or self.model_max_length

        encoded = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )

        return encoded

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        """
        Direct call to tokenizer for convenience.
        Matches HuggingFace AutoTokenizer interface.
        """
        return self.tokenizer(*args, **kwargs)
