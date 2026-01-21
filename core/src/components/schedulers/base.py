# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Base Scheduler

Abstract base class for diffusion schedulers with shared Euler stepping logic.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor


class BaseScheduler(ABC):
    """
    Abstract base class for diffusion schedulers.

    Provides shared Euler integration stepping logic and common properties.
    Subclasses must implement `set_timesteps()` to define their sigma schedule.
    """

    def __init__(self, model_path: Path, device: str):
        """
        Initialize the scheduler.

        Args:
            model_path: Path to the scheduler config directory.
            device: Device to use for tensors (e.g., "cuda" or "cpu").
        """
        self.model_path = Path(model_path) / "scheduler_config.json"
        self.device = device

        with self.model_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self._step_index = 0
        self._sigmas = None
        self._timesteps = None

    @abstractmethod
    def set_timesteps(self, step_count: int, **kwargs) -> None:
        """
        Set the sigma schedule for the given number of steps.

        Must be implemented by subclasses. Should populate:
        - self._sigmas: Tensor of sigmas with terminal zero appended
        - self._timesteps: Tensor of timesteps for logging/model input
        - self._step_index: Reset to 0

        Args:
            step_count: Number of diffusion steps.
            **kwargs: Additional scheduler-specific parameters.
        """
        pass

    def step(
            self,
            model_output: Tensor,
            sample: Tensor,
            return_dict: bool = True,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Perform one Euler integration step.

        Args:
            model_output: Predicted noise/velocity from the model.
            sample: Current latent sample.
            return_dict: If True, return tensor directly. If False, return tuple.

        Returns:
            Updated sample after one step.
        """
        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]

        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return prev_sample

    def reset(self) -> None:
        """
        Reset the scheduler state for reuse.

        Resets the step index to 0, allowing the scheduler to be used
        for another denoising loop without recreating the sigma schedule.
        """
        self._step_index = 0
