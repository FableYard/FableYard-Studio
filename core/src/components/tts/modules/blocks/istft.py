# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Inverse Short-Time Fourier Transform (ISTFT) Module

Custom ISTFT implementation that supports "same" padding for neural vocoding.
Standard torch.istft only supports center padding with windowing.
"""

import torch
from torch import nn, Tensor


class ISTFT(nn.Module):
    """
    Custom ISTFT implementation for neural vocoding.

    Supports "same" padding which is analogous to CNN padding, unlike
    standard torch.istft which only supports center padding with windowing.

    The NOLA (Nonzero Overlap Add) constraint is handled by trimming
    padded samples.

    Args:
        n_fft: Size of Fourier transform.
        hop_length: Distance between neighboring sliding window frames.
        win_length: Size of window frame and STFT filter.
        padding: Type of padding. Options are "center" or "same".
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same"
    ):
        super().__init__()

        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")

        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Register Hann window as buffer (not a parameter)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: Tensor) -> Tensor:
        """
        Compute Inverse STFT of a complex spectrogram.

        Args:
            spec: Complex spectrogram of shape (B, N, T) where B is batch size,
                  N is number of frequency bins, T is number of time frames.

        Returns:
            Reconstructed time-domain signal of shape (B, L) where L is
            the output signal length.
        """
        if self.padding == "center":
            # Fix edge frequency issues that can cause exploding gradients
            # when batch size < 16
            spec = spec.clone()
            spec[:, 0] = 0
            spec[:, -1] = 0

            # Use PyTorch native implementation for center padding
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True
            )

        # "same" padding implementation
        assert spec.dim() == 3, "Expected a 3D tensor as input"

        B, N, T = spec.shape
        pad = (self.win_length - self.hop_length) // 2

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add using fold operation
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Compute window envelope for normalization
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize by window envelope
        assert (window_envelope > 1e-11).all(), "Window envelope has near-zero values"
        y = y / window_envelope

        return y
