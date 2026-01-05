# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Union

import torch
from numpy import ndarray
from torch import arange, cat, float32, polar, ones_like

def get_1d_rotary_pos_embed(
    frequency_dimension: int,
    position: Union[ndarray, int],
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    frequency_dtype=float32,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        frequency_dimension (`int`): Dimension of the frequency tensor.
        position (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        frequency_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert frequency_dimension % 2 == 0

    if isinstance(position, int):
        position = arange(position)
    if isinstance(position, ndarray):
        position = torch.from_numpy(position)

    theta = theta * ntk_factor
    frequency = 1.0 / (theta ** (arange(0, frequency_dimension, 2, dtype=frequency_dtype, device=position.device) / frequency_dimension)) / linear_factor
    frequency = torch.outer(position, frequency)
    is_npu = frequency.device.type == "npu"

    if is_npu:
        frequency = frequency.float()

    cos_frequency = frequency.cos().repeat_interleave(2, dim=1, output_size=frequency.shape[1] * 2).float()  # [S, D]
    sin_frequency = frequency.sin().repeat_interleave(2, dim=1, output_size=frequency.shape[1] * 2).float()  # [S, D]

    return cos_frequency, sin_frequency
