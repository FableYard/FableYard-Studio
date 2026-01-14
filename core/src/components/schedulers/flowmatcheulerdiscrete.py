# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import json
from pathlib import Path

import torch
from torch import Tensor
from typing import Tuple, Union


class FlowMatchEulerDiscrete:
    def __init__(self, model_path: Path, device: str):
        self.model_path = Path(model_path) / "scheduler_config.json"
        self.device = device

        with self.model_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self._step_index = 0
        self._sigmas = None
        self._timesteps = None

    def set_timesteps(self, step_count: int, image_sequence_length: int):
        eps = 1e-5

        # Base FlowMatch schedule (probability space)
        sigmas = torch.linspace(
            1.0 - eps,
            eps,
            step_count,
            device=self.device,
            dtype=torch.float32,
            )

        # === Sequence-length-dependent shift (logit space) ===
        # Use defaults from diffusers FlowMatchEulerDiscreteScheduler
        max_shift = self.config.get("max_shift", 1.15)
        base_shift = self.config.get("base_shift", 0.5)
        max_seq = self.config.get("max_image_seq_len", 4096)
        base_seq = self.config.get("base_image_seq_len", 256)

        slope = (max_shift - base_shift) / (max_seq - base_seq)
        shift = image_sequence_length * slope + (base_shift - slope * base_seq)

        # Convert σ → logit(σ), shift, convert back
        logits = torch.log(sigmas) - torch.log1p(-sigmas)
        logits = logits + shift
        sigmas = torch.sigmoid(logits)

        # HARD invariant enforcement
        sigmas = sigmas.clamp(min=eps, max=1 - eps)

        # Append terminal zero for Euler step
        self._sigmas = torch.cat(
            [sigmas, torch.zeros(1, device=self.device)]
        )

        self._step_index = 0

        # Optional: expose pseudo-timesteps for logging
        self._timesteps = sigmas * self.config["num_train_timesteps"]

    def step(
            self,
            model_output: Tensor,
            sample: Tensor,
            return_dict: bool = True,
    ) -> Union[Tensor, Tuple[Tensor]]:
        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]

        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return prev_sample
