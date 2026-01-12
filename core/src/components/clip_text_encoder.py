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

            print(f"Loading CLIP from checkpoint: {Path(checkpoint_path).name}")

            # Try to load with text_encoder prefix (may not exist in all checkpoints)
            state_dict = load_state_dict_from_checkpoint(
                checkpoint_path=checkpoint_path,
                key_prefix="text_encoder.",
                strip_prefix=True,
                device="cpu"
            )

            if len(state_dict) == 0:
                # CLIP not in checkpoint, use default
                print("  CLIP not found in checkpoint, skipping")
                self.clip = None
                return

            # Load model structure from config
            from transformers import CLIPConfig
            # Use default CLIP-L config
            config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
            # Use module-level CLIPTextModel import
            self.clip = CLIPTextModel(config)
            self.clip.load_state_dict(state_dict, strict=False)
        else:
            # Load from directory (diffusers format)
            self.model_path = Path(component_path)
            clip_encoder_path = self.model_path
            self.clip = CLIPTextModel.from_pretrained(str(clip_encoder_path), local_files_only=True)

        # Load adapters to CPU (lazy)
        patcher = None
        if self.adapters:
            from components.adapters.adapter_patcher import AdapterPatcher

            print(f"Applying {len(self.adapters)} adapters to CLIP text encoder...")
            patcher = AdapterPatcher(model_type="clip")

            for adapter_name, adapter_info in self.adapters.items():
                adapter_path = adapter_info["path"]
                strength = adapter_info.get("strength", 1.0)
                patcher.add_adapter(adapter_path, strength)

        # Move model to device
        if self.clip is not None:
            self.clip.to(self.device)
            self.clip.eval()

        # Apply patches (computes deltas on-demand on GPU)
        if patcher is not None:
            patcher.apply_patches(self.clip.state_dict())
            del patcher
            print("CLIP adapters applied successfully.")

            config = self.clip.config
            self.hidden_size = config.hidden_size
            self.max_position_embeddings = config.max_position_embeddings
        else:
            # Defaults if no CLIP in checkpoint
            self.hidden_size = 768
            self.max_position_embeddings = 77

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
