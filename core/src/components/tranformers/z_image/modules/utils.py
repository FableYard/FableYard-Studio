# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import List, Tuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from components.tranformers.z_image.modules.constants import SEQ_MULTI_OF, X_PAD_DIM


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )

def create_coordinate_grid(size, start=None, device=None):
    if start is None:
        start = (0 for _ in size)
    axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1)

def unpatchify(
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        out_channels: int,
) -> List[torch.Tensor]:
    """Unpatchify image latents back to spatial format."""
    pH = pW = patch_size
    pF = f_patch_size
    bsz = len(x)
    assert len(size) == bsz

    # Simple unpatchify
    for i in range(bsz):
        F, H, W = size[i]
        ori_len = (F // pF) * (H // pH) * (W // pW)
        # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
        x[i] = (
            x[i][:ori_len]
            .view(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(out_channels, F, H, W)
        )
    return x

def _patchify_image(image: torch.Tensor, patch_size: int, f_patch_size: int):
    """Patchify a single image tensor: (C, F, H, W) -> (num_patches, patch_dim)."""
    pH, pW, pF = patch_size, patch_size, f_patch_size
    C, F, H, W = image.size()
    F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
    image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
    image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
    return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

def _pad_with_ids(
    feat: torch.Tensor,
    pos_grid_size: Tuple,
    pos_start: Tuple,
    device: torch.device,
    noise_mask_val: Optional[int] = None,
):
    """Pad feature to SEQ_MULTI_OF, create position IDs and pad mask."""
    ori_len = len(feat)
    pad_len = (-ori_len) % SEQ_MULTI_OF
    total_len = ori_len + pad_len

    # Pos IDs
    ori_pos_ids = create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
    if pad_len > 0:
        pad_pos_ids = (
            create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
            .flatten(0, 2)
            .repeat(pad_len, 1)
        )
        pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
        padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
        pad_mask = torch.cat(
            [
                torch.zeros(ori_len, dtype=torch.bool, device=device),
                torch.ones(pad_len, dtype=torch.bool, device=device),
            ]
        )
    else:
        pos_ids = ori_pos_ids
        padded_feat = feat
        pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

    noise_mask = [noise_mask_val] * total_len if noise_mask_val is not None else None  # token level
    return padded_feat, pos_ids, pad_mask, total_len, noise_mask

def patchify_and_embed(
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int
):
    """Patchify for basic mode: single image per batch item."""
    device = all_image[0].device
    all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
    all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

    for image, cap_feat in zip(all_image, all_cap_feats):
        # Caption
        cap_out, cap_pos_ids, cap_pad_mask, cap_len, _ = _pad_with_ids(
            cap_feat, (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0), device
        )
        all_cap_out.append(cap_out)
        all_cap_pos_ids.append(cap_pos_ids)
        all_cap_pad_mask.append(cap_pad_mask)

        # Image
        img_patches, size, (F_t, H_t, W_t) = _patchify_image(image, patch_size, f_patch_size)
        img_out, img_pos_ids, img_pad_mask, _, _ = _pad_with_ids(
            img_patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device
        )
        all_img_out.append(img_out)
        all_img_size.append(size)
        all_img_pos_ids.append(img_pos_ids)
        all_img_pad_mask.append(img_pad_mask)

    return (
        all_img_out,
        all_cap_out,
        all_img_size,
        all_img_pos_ids,
        all_cap_pos_ids,
        all_img_pad_mask,
        all_cap_pad_mask,
    )

def build_unified_sequence(
    x: torch.Tensor,
    x_freqs: torch.Tensor,
    x_seqlens: List[int],
    cap: torch.Tensor,
    cap_freqs: torch.Tensor,
    cap_seqlens: List[int],
    device: torch.device,
):
    bsz = len(x_seqlens)
    unified = []
    unified_frequencies = []

    for i in range(bsz):
        x_len, cap_len = x_seqlens[i], cap_seqlens[i]
        unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
        unified_frequencies.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

    unified_sequence_length = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
    max_sequence_length = max(unified_sequence_length)

    unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
    unified_frequencies = pad_sequence(unified_frequencies, batch_first=True, padding_value=0.0)

    attention_mask = torch.zeros((bsz, max_sequence_length), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_sequence_length):
        attention_mask[i, :seq_len] = 1

    return unified, unified_frequencies, attention_mask