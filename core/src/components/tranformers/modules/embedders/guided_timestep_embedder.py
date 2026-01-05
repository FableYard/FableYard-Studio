# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch.nn import Module

from components.tranformers.modules.embedders.timestep_embedder import TimestepEmbedding
from components.tranformers.modules.projectors.pixart_alpha_text_projector import PixArtAlphaTextProjection
from components.tranformers.modules.timesteps import Timesteps
from utils import info


class CombinedTimestepGuidanceTextProjEmbeddings(Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(channel_count=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(input_channel_count=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(input_channel_count=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        info(f"Timestep: {timestep}")
        timesteps_proj = self.time_proj(timestep)

        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj)  # (N, D)
        time_guidance_emb = timesteps_emb + guidance_emb
        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = time_guidance_emb + pooled_projections

        return conditioning