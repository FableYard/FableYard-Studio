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

    def __init__(
        self,
        component_path: str | Path = None,
        device: str | torch.device = None,
        adapters: dict = None,
        checkpoint_path: str | Path = None
    ):
        self.device = torch.device(device)
        self.adapters = adapters or {}

        if checkpoint_path is not None:
            # Load from single checkpoint file
            from utils.checkpoint_utils import load_state_dict_from_checkpoint

            print(f"Loading T5 from checkpoint: {Path(checkpoint_path).name}")

            # Try to load with text_encoder_2 prefix (may not exist in all checkpoints)
            state_dict = load_state_dict_from_checkpoint(
                checkpoint_path=checkpoint_path,
                key_prefix="text_encoder_2.",
                strip_prefix=True,
                device="cpu"
            )

            if len(state_dict) == 0:
                # T5 not in checkpoint, use default
                print("  T5 not found in checkpoint, skipping")
                self.t5 = None
                return

            # Load model structure from config
            from transformers import T5Config
            # Use default T5-XXL config
            config = T5Config.from_pretrained("google/t5-v1_1-xxl")
            self.t5 = T5EncoderModel(config)
            self.t5.load_state_dict(state_dict, strict=False)
        else:
            # Load from directory (diffusers format)
            self.model_path = Path(component_path)
            t5_encoder_path = self.model_path
            self.t5 = T5EncoderModel.from_pretrained(str(t5_encoder_path))

        # Load adapters to CPU (lazy)
        patcher = None
        if self.adapters:
            from components.adapters.adapter_patcher import AdapterPatcher

            print(f"Applying {len(self.adapters)} adapters to T5 text encoder...")
            patcher = AdapterPatcher(model_type="t5")

            for adapter_name, adapter_info in self.adapters.items():
                adapter_path = adapter_info["path"]
                strength = adapter_info.get("strength", 1.0)
                patcher.add_adapter(adapter_path, strength)

        # Move model to device
        if self.t5 is not None:
            self.t5.to(self.device)
            self.t5.eval()

        # Apply patches (computes deltas on-demand on GPU)
        if patcher is not None:
            patcher.apply_patches(self.t5.state_dict())
            del patcher
            print("T5 adapters applied successfully.")

            config = self.t5.config
            self.hidden_size = config.d_model        # 4096
            self.num_layers = config.num_layers      # 24
            self.num_heads = config.num_heads        # 64
        else:
            # Defaults if no T5 in checkpoint
            self.hidden_size = 4096
            self.num_layers = 24
            self.num_heads = 64

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
