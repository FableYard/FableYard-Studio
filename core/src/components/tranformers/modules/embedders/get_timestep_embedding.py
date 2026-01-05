# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from math import log as math_log

from torch import Tensor, arange, float32, exp, cat, sin, cos
from torch.nn import functional

def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dimension: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dimension = embedding_dimension // 2
    exponent = -math_log(max_period) * arange(
        start=0, end=half_dimension, dtype=timesteps.dtype, device=timesteps.device
    )
    exponent = exponent / (half_dimension - downscale_freq_shift)

    embedding = exp(exponent)
    embedding = timesteps[:, None] * embedding[None, :]

    # scale embeddings
    embedding = scale * embedding

    # concat sine and cosine embeddings
    embedding = cat([sin(embedding), cos(embedding)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        embedding = cat([embedding[:, half_dimension:], embedding[:, :half_dimension]], dim=-1)

    # zero pad
    if embedding_dimension % 2 == 1:
        embedding = functional.pad(embedding, (0, 1, 0, 0))
    return embedding
