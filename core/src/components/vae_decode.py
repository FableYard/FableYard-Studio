# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from typing import Tuple
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file

from components.blocks import MidBlock, UpDecoderBlock2D
from components.tranformers.modules.activators.utils import get_activation


class KullbackLeibler(nn.Module):
    """
    Config-driven AutoencoderKL (decoder-only) implementation.
    Architecture and behavior are fully defined by vae/config.json.
    """

    def __init__(self, model_path: Path | str, device: str):
        super().__init__()
        self.model_path = Path(model_path)
        self.device = device

        # populated in load()
        self._scaling_factor = 1.0
        self._shift_factor = 0.0

        self.conv_in = None
        self.mid_block = None
        self.up_blocks = None
        self.conv_norm_out = None
        self.conv_act = None
        self.conv_out = None
        self.post_quant_conv = None

    def load(self) -> None:
        """Load VAE decoder architecture and weights from config + safetensors."""
        cfg_path = self.model_path / "config.json"
        weights_path = self.model_path / "diffusion_pytorch_model.safetensors"

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # ---- authoritative config values ----
        latent_channels = cfg["latent_channels"]
        out_channels = cfg["out_channels"]
        block_out_channels = cfg["block_out_channels"]
        layers_per_block = cfg["layers_per_block"]
        norm_num_groups = cfg["norm_num_groups"]
        act_fn = cfg["act_fn"]
        up_block_types = cfg["up_block_types"]
        mid_block_add_attention = cfg["mid_block_add_attention"]
        use_post_quant_conv = cfg["use_post_quant_conv"]

        self._scaling_factor = cfg.get("scaling_factor", 1.0)
        self._shift_factor = cfg.get("shift_factor", 0.0)

        # ---- build decoder on meta device ----
        with torch.device("meta"):
            self.conv_in = nn.Conv2d(
                latent_channels,
                block_out_channels[-1],
                kernel_size=3,
                padding=1,
            )

            self.mid_block = MidBlock(
                input_channel_count=block_out_channels[-1],
                dropout=0.0,
                layer_count=1,
                resnet_group_count=norm_num_groups,
                resnet_activation_function=act_fn,
                add_attention=mid_block_add_attention,
            )

            self.up_blocks = nn.ModuleList()
            reversed_channels = list(reversed(block_out_channels))

            prev_channels = reversed_channels[0]
            for i in range(len(up_block_types)):
                out_ch = reversed_channels[i]
                is_final = i == len(up_block_types) - 1

                self.up_blocks.append(
                    UpDecoderBlock2D(
                        input_channel_count=prev_channels,
                        output_channel_count=out_ch,
                        layer_count=layers_per_block + 1,
                        dropout=0.0,
                        add_upsample=not is_final,
                        group_count=norm_num_groups,
                        activation_function=act_fn,
                    )
                )
                prev_channels = out_ch

            self.conv_norm_out = nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                eps=1e-6,
            )
            self.conv_act = get_activation(act_fn)
            self.conv_out = nn.Conv2d(
                block_out_channels[0],
                out_channels,
                kernel_size=3,
                padding=1,
            )

            if use_post_quant_conv:
                self.post_quant_conv = nn.Conv2d(
                    latent_channels,
                    latent_channels,
                    kernel_size=1,
                )

        # ---- materialize + load weights ----
        self.to_empty(device=self.device)

        state = load_file(str(weights_path), device=self.device)
        cleaned_state = {
            k[len("decoder."):] if k.startswith("decoder.") else k: v
            for k, v in state.items()
        }

        self.load_state_dict(cleaned_state, strict=False, assign=True)
        self.eval()

        del state, cleaned_state

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor into image space.

        Args:
            z: (B, latent_channels, H, W)

        Returns:
            (B, out_channels, H*8, W*8)
        """
        # Validate input
        assert not torch.isnan(z).any(), \
            f"Input to VAE decode contains NaN! Shape: {z.shape}, dtype: {z.dtype}"
        assert not torch.isinf(z).any(), \
            f"Input to VAE decode contains Inf! Shape: {z.shape}, dtype: {z.dtype}"

        z = (z / self._scaling_factor) + self._shift_factor

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
            assert not torch.isnan(z).any(), \
                f"Output of post_quant_conv contains NaN! Shape: {z.shape}"
            assert not torch.isinf(z).any(), \
                f"Output of post_quant_conv contains Inf! Shape: {z.shape}"

        x = self.conv_in(z)
        assert not torch.isnan(x).any(), \
            f"Output of conv_in contains NaN! Shape: {x.shape}"
        assert not torch.isinf(x).any(), \
            f"Output of conv_in contains Inf! Shape: {x.shape}"

        x = self.mid_block(x)
        assert not torch.isnan(x).any(), \
            f"Output of mid_block contains NaN! Shape: {x.shape}"
        assert not torch.isinf(x).any(), \
            f"Output of mid_block contains Inf! Shape: {x.shape}"

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            assert not torch.isnan(x).any(), \
                f"Output of up_block[{i}] contains NaN! Shape: {x.shape}"
            assert not torch.isinf(x).any(), \
                f"Output of up_block[{i}] contains Inf! Shape: {x.shape}"

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        # Validate final output
        assert not torch.isnan(x).any(), \
            f"Final VAE output contains NaN! Shape: {x.shape}"
        assert not torch.isinf(x).any(), \
            f"Final VAE output contains Inf! Shape: {x.shape}"

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(decoder-only)"
