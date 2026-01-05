# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
CLIP Tokenizer Component
Handles CLIP tokenization for pipelines with ByteLevelBPE.
"""
import json
from pathlib import Path
from typing import List

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from utils import info


class CLIPTokenizer:
    """
    CLIP tokenizer using ByteLevelBPE.
    """

    def __init__(self, component_path: Path, device: str):
        self.device = device
        self._component_path = Path(component_path)

        with open(Path(component_path, "tokenizer_config.json"), "r") as f:
            self._config = json.load(f)

        vocab = self._component_path / "vocab.json"
        merges = self._component_path / "merges.txt"

        tokenizer = Tokenizer(
            BPE.from_file(str(vocab), str(merges))
        )

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        self._tokenizer = tokenizer

        self._max_length = self._config["model_max_length"]
        self._bos_token_id = self._config["bos_token"]
        self._eos_token_id = self._config["eos_token"]
        self._pad_token_id = self._eos_token_id
        self._unk_token_id = self._eos_token_id

        self._special_ids = {
            self._bos_token_id,
            self._eos_token_id,
            self._pad_token_id,
        }

    def encode(
        self,
        text: str | List[str],
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: bool = False,
    ) -> torch.Tensor:
        tokenizer = self._tokenizer

        tokenizer.no_truncation()
        tokenizer.no_padding()

        if truncation:
            tokenizer.enable_truncation(self._max_length)

        if padding:
            tokenizer.enable_padding(length=self._max_length, pad_id=49407)

        if isinstance(text, list):
            encodings = tokenizer.encode_batch(text)
            ids = [e.ids for e in encodings]
        else:
            ids = [tokenizer.encode(text, add_special_tokens=add_special_tokens).ids]

        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in self._special_ids]

        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    # def batch_encode(
    #     self,
    #     texts: List[str],
    #     add_special_tokens: bool = True,
    #     max_length: Optional[int] = None,
    #     truncation: bool = True,
    #     padding: bool = True
    # ) -> List[List[int]]:
    #     """
    #     Encode batch of texts to token IDs.
    #
    #     Args:
    #         texts: List of input texts to tokenize
    #         add_special_tokens: Whether to add BOS/EOS tokens
    #         max_length: Maximum sequence length (defaults to 77)
    #         truncation: Whether to truncate sequences exceeding max_length
    #         padding: Whether to pad sequences to max_length
    #
    #     Returns:
    #         List of token ID lists
    #     """
    #     max_len = max_length or self._max_length
    #
    #     # Configure encoding
    #     self._tokenizer.enable_truncation(max_len if truncation else None)
    #     if padding:
    #         self._tokenizer.enable_padding(length=max_len, pad_id=self._pad_token_id)
    #     else:
    #         self._tokenizer.no_padding()
    #
    #     # Encode batch
    #     encodings = self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
    #     return [enc.ids for enc in encodings]
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
    #     return [self.decode(token_ids, skip_special_tokens) for token_ids in batch_token_ids]
