# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FlowMatch Euler Discrete Scheduler

Scheduler using FlowMatch sigma schedule with Euler integration.
"""

from pathlib import Path

import torch

from .base import BaseScheduler
from .sigma_schedules import flow_match_schedule, apply_logit_shift


class FlowMatchEulerDiscrete(BaseScheduler):
    """
    FlowMatch scheduler with sequence-length-dependent logit shift.

    Uses a linear schedule in probability space with a shift applied
    in logit space based on the image sequence length.
    """

    def set_timesteps(self, step_count: int, image_sequence_length: int) -> None:
        """
        Set the FlowMatch sigma schedule with logit shift.

        Args:
            step_count: Number of diffusion steps.
            image_sequence_length: Latent sequence length for shift calculation.
        """
        eps = 1e-5

        # Base FlowMatch schedule (probability space)
        sigmas = flow_match_schedule(step_count, device=self.device, eps=eps)

        # === Sequence-length-dependent shift (logit space) ===
        # Use defaults from diffusers FlowMatchEulerDiscreteScheduler
        max_shift = self.config.get("max_shift", 1.15)
        base_shift = self.config.get("base_shift", 0.5)
        max_seq = self.config.get("max_image_seq_len", 4096)
        base_seq = self.config.get("base_image_seq_len", 256)

        slope = (max_shift - base_shift) / (max_seq - base_seq)
        shift = image_sequence_length * slope + (base_shift - slope * base_seq)

        # Apply logit shift
        sigmas = apply_logit_shift(sigmas, shift, eps)

        # Append terminal zero for Euler step
        self._sigmas = torch.cat(
            [sigmas, torch.zeros(1, device=self.device)]
        )

        self._step_index = 0

        # Optional: expose pseudo-timesteps for logging
        self._timesteps = sigmas * self.config["num_train_timesteps"]
