# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Linear Quadratic Euler Scheduler

Scheduler using a linear-quadratic sigma schedule with Euler integration.
This matches ComfyUI's "DDIM" sampler behavior (which is actually Euler stepping
with a different sigma schedule).
"""

from pathlib import Path
from typing import Optional

from .base import BaseScheduler
from .sigma_schedules import linear_quadratic_schedule


class LinearQuadraticEuler(BaseScheduler):
    """
    Scheduler with linear-quadratic sigma schedule and Euler stepping.

    The linear-quadratic schedule produces crisp results with Flux models
    by using a linear phase for fine detail refinement followed by a
    quadratic phase for main denoising.
    """

    def __init__(
            self,
            model_path: Path,
            device: str,
            threshold_noise: float = 0.025,
            linear_steps: Optional[int] = None
    ):
        """
        Initialize the Linear Quadratic Euler scheduler.

        Args:
            model_path: Path to the scheduler config directory.
            device: Device to use for tensors (e.g., "cuda" or "cpu").
            threshold_noise: Noise level at which linear phase ends (default 0.025).
            linear_steps: Number of steps in linear phase (default: steps // 2).
        """
        super().__init__(model_path, device)
        self.threshold_noise = threshold_noise
        self.default_linear_steps = linear_steps

    def set_timesteps(
            self,
            step_count: int,
            image_sequence_length: Optional[int] = None,
            linear_steps: Optional[int] = None
    ) -> None:
        """
        Set the linear-quadratic sigma schedule.

        Args:
            step_count: Number of diffusion steps.
            image_sequence_length: Ignored (included for API compatibility).
            linear_steps: Override for linear phase steps (default: step_count // 2).
        """
        # Use provided linear_steps, or instance default, or half of step_count
        actual_linear_steps = linear_steps or self.default_linear_steps or (step_count // 2)

        # Generate linear-quadratic schedule (already includes terminal zero)
        self._sigmas = linear_quadratic_schedule(
            steps=step_count,
            sigma_max=1.0,
            threshold_noise=self.threshold_noise,
            linear_steps=actual_linear_steps,
            device=self.device
        )

        self._step_index = 0

        # Expose pseudo-timesteps for logging (excluding terminal zero)
        self._timesteps = self._sigmas[:-1] * self.config["num_train_timesteps"]
