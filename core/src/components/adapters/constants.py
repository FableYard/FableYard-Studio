# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from components.adapters.adapter_role import AdapterRole

# Flux1 Dev
DOUBLE_STREAM_KEY_MAP = {
    # ---- latent self-attention ----
    "attn.to_q": AdapterRole.ATTENTION,
    "attn.to_k": AdapterRole.ATTENTION,
    "attn.to_v": AdapterRole.ATTENTION,
    "attn.to_out.0": AdapterRole.ATTENTION,
    "attention.to_q.lora_A": AdapterRole.ATTENTION,
    "attention.to_q.lora_B": AdapterRole.ATTENTION,
    "attention.to_k.lora_A": AdapterRole.ATTENTION,
    "attention.to_k.lora_B": AdapterRole.ATTENTION,
    "attention.to_v.lora_A": AdapterRole.ATTENTION,
    "attention.to_v.lora_B": AdapterRole.ATTENTION,
    "attention.to_out.0.lora_A": AdapterRole.ATTENTION,
    "attention.to_out.0.lora_B": AdapterRole.ATTENTION,
    "lora_unet_double_blocks_0_img_attn_proj": AdapterRole.ATTENTION,
    "lora_unet_double_blocks_0_img_attn_qkv": AdapterRole.ATTENTION,
    "lora_unet_double_blocks_0_txt_attn_proj": AdapterRole.ATTENTION,
    "lora_unet_double_blocks_0_txt_attn_qkv": AdapterRole.ATTENTION,

    "attn.norm_q": AdapterRole.NORM_DANGEROUS,
    "attn.norm_k": AdapterRole.NORM_DANGEROUS,

    # ---- conditioning (added) attention ----
    "attn.add_q_proj": AdapterRole.ATTENTION_CONTEXT,
    "attn.add_k_proj": AdapterRole.ATTENTION_CONTEXT,
    "attn.add_v_proj": AdapterRole.ATTENTION_CONTEXT,
    "attn.to_add_out": AdapterRole.ATTENTION_CONTEXT,

    "attn.norm_added_q": AdapterRole.ADALN_CONTEXT,
    "attn.norm_added_k": AdapterRole.ADALN_CONTEXT,

    # ---- feed-forward networks ----
    "ff.net.0.proj": AdapterRole.FFN,
    "ff.net.2": AdapterRole.FFN,
    "feed_forward.w1.lora_A": AdapterRole.FFN,
    "feed_forward.w1.lora_B": AdapterRole.FFN,
    "feed_forward.w2.lora_A": AdapterRole.FFN,
    "feed_forward.w2.lora_B": AdapterRole.FFN,
    "feed_forward.w3.lora_A": AdapterRole.FFN,
    "feed_forward.w3.lora_B": AdapterRole.FFN,
    "lora_unet_double_blocks_0_img_mlp_0": AdapterRole.FFN,
    "lora_unet_double_blocks_0_img_mlp_2": AdapterRole.FFN,
    "lora_unet_double_blocks_0_txt_mlp_0": AdapterRole.FFN,
    "lora_unet_double_blocks_0_txt_mlp_2": AdapterRole.FFN,

    "ff_context.net.0.proj": AdapterRole.FFN_CONTEXT,
    "ff_context.net.2": AdapterRole.FFN_CONTEXT,

    # ---- adaptive normalization ----
    "adaLN_modulation.0.lora_A": AdapterRole.ADALN,
    "adaLN_modulation.0.lora_B": AdapterRole.ADALN,
    "norm1.linear": AdapterRole.ADALN,
    "lora_unet_double_blocks_0_img_mod_lin": AdapterRole.ADALN,
    "lora_unet_double_blocks_0_txt_mod_lin": AdapterRole.ADALN,
    "norm1_context.linear": AdapterRole.ADALN_CONTEXT,
}


# ----------------------------------
# Single-stream transformer blocks
# single_transformer_blocks.{i}.*
# ----------------------------------

# Flux1 Dev
SINGLE_STREAM_KEY_MAP = {
    # ---- merged attention ----
    "attn.to_q": AdapterRole.ATTENTION,
    "attn.to_k": AdapterRole.ATTENTION,
    "attn.to_v": AdapterRole.ATTENTION,

    "attn.norm_q": AdapterRole.NORM_DANGEROUS,
    "attn.norm_k": AdapterRole.NORM_DANGEROUS,

    # ---- merged adaptive norm ----
    "norm.linear": AdapterRole.ADALN,
    "lora_unet_single_blocks_0_modulation_lin": AdapterRole.ADALN,

    # ---- collapsed MLP ----
    "proj_mlp": AdapterRole.FFN,
    "proj_out": AdapterRole.FFN,
    "lora_unet_single_blocks_0_linear1": AdapterRole.FFN,
    "lora_unet_single_blocks_0_linear2": AdapterRole.FFN,
}