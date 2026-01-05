# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import CLIPTextModel


class CLIPTextEncoder:
    """
    CLIP text encoder using HuggingFace Transformers.
    """

    def __init__(self, component_path: str | Path, device: str | torch.device):
        self.device = torch.device(device)
        self.model_path = Path(component_path)

        clip_encoder_path = self.model_path

        self.clip = CLIPTextModel.from_pretrained(str(clip_encoder_path), local_files_only=True)
        self.clip.to(self.device)
        self.clip.eval()

        config = self.clip.config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

    @torch.no_grad()
    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_pooled(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.pooler_output
