# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
T5 Tokenizer Component
Handles T5 tokenization using SentencePiece.
"""
import json
from os import path as os_path
from pathlib import Path
from typing import List, Optional
import torch

from sentencepiece import SentencePieceProcessor

class T5Tokenizer:
    """
    T5 tokenizer component using SentencePiece.

    Combines tokenization logic with pipeline lifecycle management.
    Loads from a local .spm model file and provides full tokenization
    capabilities within the pipeline context.
    """
    def __init__(self, component_path: Path, device: str) -> None:
        self._component_path = component_path
        with open(os_path.join(self._component_path, "tokenizer_config.json"), "r") as f:
            self._config = json.load(f)

        self.sp_model = None
        self.extra_ids = 0
        self._max_length = 0
        self._vocab_size = 0
        self._eos_token_id = 1
        self._unk_token_id = 2
        self._pad_token_id = 0
        self._bos_token_id = None

    def load(self) -> None:
        """Load T5 tokenizer from the SentencePiece model file"""
        t5_path = Path(self._component_path, "spiece.model")

        # Load SentencePiece model
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(str(t5_path))

        # T5-specific configuration
        self.extra_ids = 100
        self._max_length = 512
        self._vocab_size = self.sp_model.vocab_size()

        # T5 special token IDs
        self._eos_token_id = 1  # </s>
        self._unk_token_id = 2  # <unk>
        self._pad_token_id = 0  # <pad>
        self._bos_token_id = None  # T5 doesn't use BOS

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False
    ) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add EOS token
            max_length: Maximum sequence length (defaults to 512)
            truncation: Whether to truncate sequences exceeding max_length
            padding: Whether to pad sequences to max_length

        Returns:
            List of token IDs
        """
        max_len = max_length or self._max_length

        # Encode text using SentencePiece
        tokens = self.sp_model.EncodeAsIds(text)

        # Add EOS token if requested
        if add_special_tokens:
            tokens.append(self._eos_token_id)

        # Truncate if needed
        if truncation and len(tokens) > max_len:
            tokens = tokens[:max_len]

        # Pad if needed
        if padding and len(tokens) < max_len:
            tokens.extend([self._pad_token_id] * (max_len - len(tokens)))

        tokens = torch.tensor([tokens])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to remove special tokens from output

        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Filter out special tokens (pad, eos, unk)
            token_ids = [
                tid for tid in token_ids
                if tid not in (self._pad_token_id, self._eos_token_id)
            ]

        return self.sp_model.DecodeIds(token_ids)

    def get_extra_id(self, n: int) -> int:
        """
        Get token ID for <extra_id_N>.

        Args:
            n: Extra ID number (0 to extra_ids-1)

        Returns:
            Token ID for <extra_id_N>
        """
        if n >= self.extra_ids:
            raise ValueError(f"Extra ID {n} exceeds maximum {self.extra_ids - 1}")

        # Extra IDs are indexed from the end of vocabulary
        return self._vocab_size - 1 - n

    # def batch_encode(
    #     self,
    #     texts: List[str],
    #     add_special_tokens: bool = True,
    #     max_length: Optional[int] = None,
    #     truncation: bool = True,
    #     padding: bool = True
    # ) -> List[torch.Tensor]:
    #     """
    #     Encode batch of texts to token IDs.
    #
    #     Args:
    #         texts: List of input texts to tokenize
    #         add_special_tokens: Whether to add EOS token
    #         max_length: Maximum sequence length (defaults to 512)
    #         truncation: Whether to truncate sequences exceeding max_length
    #         padding: Whether to pad sequences to max_length
    #
    #     Returns:
    #         List of token ID lists
    #     """
    #     return [
    #         self.encode(
    #             text,
    #             add_special_tokens=add_special_tokens,
    #             max_length=max_length,
    #             truncation=truncation,
    #             padding=padding
    #         )
    #         for text in texts
    #     ]
    #
    # def batch_decode(
    #     self,
    #     batch_token_ids: List[List[int]],
    #     skip_special_tokens: bool = True
    # ) -> List[str]:
    #     """
    #     Decode batch of token IDs to texts.
    #
    #     Args:
    #         batch_token_ids: List of token ID lists to decode
    #         skip_special_tokens: Whether to remove special tokens from output
    #
    #     Returns:
    #         List of decoded text strings
    #     """
    #     return [
    #         self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    #         for token_ids in batch_token_ids
    #     ]