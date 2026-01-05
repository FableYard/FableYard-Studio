# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import T5EncoderModel


class T5TextEncoder:
    """
    T5 text encoder using HuggingFace Transformers.
    """

    def __init__(self, component_path: str | Path, device: str | torch.device):
        self.device = torch.device(device)
        self.model_path = Path(component_path)

        t5_encoder_path = self.model_path

        self.t5 = T5EncoderModel.from_pretrained(str(t5_encoder_path))
        self.t5.to(self.device)
        self.t5.eval()

        config = self.t5.config
        self.hidden_size = config.d_model        # 4096
        self.num_layers = config.num_layers      # 24
        self.num_heads = config.num_heads        # 64

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

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        return outputs.last_hidden_state
