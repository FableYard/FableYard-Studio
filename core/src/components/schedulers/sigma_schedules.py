# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Sigma Schedules

Standalone sigma schedule generation functions for diffusion models.
"""

import torch
from torch import Tensor


def linear_quadratic_schedule(
        steps: int,
        sigma_max: float = 1.0,
        threshold_noise: float = 0.025,
        linear_steps: int = None,
        device: str = "cpu"
) -> Tensor:
    """
    Generate a linear-quadratic sigma schedule (from ComfyUI).

    Creates a schedule that starts with a linear phase for fine detail refinement,
    then transitions to a quadratic phase for the main denoising process.

    Args:
        steps: Total number of diffusion steps.
        sigma_max: Maximum sigma value (typically 1.0).
        threshold_noise: Noise level at which linear phase ends (default 0.025).
        linear_steps: Number of steps in linear phase (default: steps // 2).
        device: Device for the output tensor.

    Returns:
        Tensor of shape (steps + 1,) with sigmas from high to low noise,
        ending with a terminal zero for Euler stepping.
    """
    if linear_steps is None:
        linear_steps = steps // 2

    # Linear phase: [0, threshold_noise] over first half of steps
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]

    # Quadratic phase: smooth transition to 1.0
    quadratic_steps = steps - linear_steps
    threshold_noise_step_diff = linear_steps - threshold_noise * steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)

    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, steps)
    ]

    # Combine and invert (high to low noise)
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]

    sigmas = torch.FloatTensor(sigma_schedule).to(device) * sigma_max

    # Append terminal zero for Euler step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    return sigmas


def flow_match_schedule(
        steps: int,
        device: str = "cpu",
        eps: float = 1e-5
) -> Tensor:
    """
    Generate a FlowMatch sigma schedule.

    Creates a linear schedule in probability space from (1-eps) to eps.

    Args:
        steps: Number of diffusion steps.
        device: Device for the output tensor.
        eps: Small epsilon to avoid exact 0 or 1 values.

    Returns:
        Tensor of shape (steps,) with sigmas from high to low noise.
    """
    return torch.linspace(
        1.0 - eps,
        eps,
        steps,
        device=device,
        dtype=torch.float32,
    )


def apply_logit_shift(
        sigmas: Tensor,
        shift: float,
        eps: float = 1e-5
) -> Tensor:
    """
    Apply a logit-space shift to sigmas (sequence-length-dependent).

    Converts sigmas to logit space, applies additive shift, converts back,
    and enforces numerical stability constraints.

    Args:
        sigmas: Input sigma tensor in [0, 1].
        shift: Shift amount in logit space.
        eps: Small epsilon for numerical stability.

    Returns:
        Shifted sigmas clamped to [eps, 1-eps].
    """
    # Convert σ → logit(σ), shift, convert back
    logits = torch.log(sigmas) - torch.log1p(-sigmas)
    logits = logits + shift
    sigmas = torch.sigmoid(logits)

    # HARD invariant enforcement
    sigmas = sigmas.clamp(min=eps, max=1 - eps)

    return sigmas
