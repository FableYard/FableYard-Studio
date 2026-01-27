# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Vocos Vocoder

Converts audio tokens (hidden states from the encoder) to waveforms.
Uses ConvNeXt blocks followed by ISTFT for efficient audio synthesis.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn, Tensor

from components.tts.modules.blocks.convnext import ConvNeXtBlock
from components.tts.modules.blocks.istft import ISTFT


class VocosBackbone(nn.Module):
    """
    Vocos backbone built with ConvNeXt blocks.

    Args:
        input_channels: Number of input feature channels.
        dim: Hidden dimension of the model.
        intermediate_dim: Intermediate dimension for ConvNeXt blocks.
        num_layers: Number of ConvNeXt layers.
        input_kernel_size: Kernel size for input convolution.
        dw_kernel_size: Kernel size for depthwise convolutions.
        layer_scale_init_value: Initial value for layer scaling.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        input_kernel_size: int = 9,
        dw_kernel_size: int = 9,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()

        # Input embedding convolution
        self.embed = nn.Conv1d(
            input_channels, dim,
            kernel_size=input_kernel_size,
            padding=input_kernel_size // 2,
            padding_mode='zeros'
        )

        # Initial layer normalization
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Stack of ConvNeXt blocks
        scale_value = layer_scale_init_value or 1 / num_layers ** 0.5
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                dw_kernel_size=dw_kernel_size,
                layer_scale_init_value=scale_value,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, L) where B is batch size,
               C is input_channels, L is sequence length.

        Returns:
            Output tensor of shape (B, dim, L).
        """
        # Input embedding: (B, C, L) -> (B, dim, L)
        x = self.embed(x)

        # Layer norm: transpose to (B, L, dim), normalize, transpose back
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # ConvNeXt blocks
        for conv_block in self.convnext:
            x = conv_block(x)

        # Final layer norm
        x = self.final_layer_norm(x.transpose(1, 2))
        x = x.transpose(1, 2)

        return x


class ISTFTHead(nn.Module):
    """
    ISTFT head for predicting STFT complex coefficients and reconstructing audio.

    Predicts magnitude and phase components, then uses ISTFT to reconstruct
    the time-domain signal.

    Args:
        dim: Hidden dimension of the backbone.
        n_fft: FFT size.
        hop_length: Hop length for ISTFT (should align with input resolution).
        padding: Padding type for ISTFT ("center" or "same").
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "center"
    ):
        super().__init__()

        # Output projection: predict n_fft/2+1 magnitude + n_fft/2+1 phase
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)

        # ISTFT module
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding
        )

    @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, dim, L) from backbone.

        Returns:
            Reconstructed audio of shape (B, T) where T is output signal length.
        """
        # Project to magnitude and phase: (B, dim, L) -> (B, n_fft+2, L)
        x = self.out(x.transpose(1, 2)).transpose(1, 2)

        # Split into magnitude and phase
        mag, p = x.chunk(2, dim=1)

        # Magnitude: exp with clipping to prevent overflow
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)

        # Phase: convert to complex via cos/sin
        x = torch.cos(p)
        y = torch.sin(p)

        # Construct complex spectrogram: mag * e^(i*phase)
        S = mag * (x + 1j * y)

        # Reconstruct audio via ISTFT
        audio = self.istft(S)

        return audio


class VocosDecoder(nn.Module):
    """
    Complete Vocos decoder for converting hidden states to audio.

    Takes audio tokens (hidden states from the encoder) and produces
    waveform audio through:
    1. Upsampling via interpolation
    2. ConvNeXt backbone processing
    3. ISTFT-based audio reconstruction

    Default architecture matches Soprano's decoder:
    - 512-dim input (hidden states)
    - 768-dim backbone with 8 layers
    - 4x upsampling
    - 2048 FFT size, 512 hop length -> 32kHz output

    Args:
        num_input_channels: Dimension of input hidden states.
        decoder_num_layers: Number of ConvNeXt layers.
        decoder_dim: Hidden dimension of backbone.
        decoder_intermediate_dim: Intermediate dimension (default: 3x decoder_dim).
        hop_length: STFT hop length.
        n_fft: FFT size.
        upscale: Upsampling factor.
        dw_kernel: Depthwise convolution kernel size.
    """

    def __init__(
        self,
        num_input_channels: int = 512,
        decoder_num_layers: int = 8,
        decoder_dim: int = 768,
        decoder_intermediate_dim: Optional[int] = None,
        hop_length: int = 512,
        n_fft: int = 2048,
        upscale: int = 4,
        dw_kernel: int = 3,
    ):
        super().__init__()

        self.decoder_initial_channels = num_input_channels
        self.num_layers = decoder_num_layers
        self.dim = decoder_dim
        self.intermediate_dim = decoder_intermediate_dim or decoder_dim * 3
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.upscale = upscale
        self.dw_kernel = dw_kernel

        # Backbone
        self.decoder = VocosBackbone(
            input_channels=self.decoder_initial_channels,
            dim=self.dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            input_kernel_size=1,  # Soprano uses kernel_size=1 for input
            dw_kernel_size=dw_kernel,
        )

        # ISTFT head
        self.head = ISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Convert hidden states to audio waveform.

        Args:
            x: Hidden states from encoder of shape (B, C, T) where
               B is batch size, C is num_input_channels (512),
               T is number of audio tokens.

        Returns:
            Audio waveform of shape (B, samples).
        """
        T = x.size(2)

        # Upsample: linear interpolation
        # Output length: upscale * (T - 1) + 1
        x = torch.nn.functional.interpolate(
            x,
            size=self.upscale * (T - 1) + 1,
            mode='linear',
            align_corners=True
        )

        # Process through backbone
        x = self.decoder(x)

        # Reconstruct audio
        audio = self.head(x)

        return audio

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ) -> "VocosDecoder":
        """
        Load a pretrained Vocos decoder.

        Args:
            model_path: Path to the model directory containing decoder.pth.
            device: Device to load the model to.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Loaded VocosDecoder instance.
        """
        model_path = Path(model_path)
        decoder_path = model_path / "decoder.pth"

        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder weights not found at {decoder_path}")

        # Create decoder with default Soprano architecture
        decoder = cls(**kwargs)

        # Load weights
        state_dict = torch.load(decoder_path, map_location=device, weights_only=True)
        decoder.load_state_dict(state_dict)

        decoder.to(device)
        decoder.eval()

        return decoder
