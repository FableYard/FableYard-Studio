# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch.nn import Module

from components.tranformers.modules.embedders.label import LabelEmbedding
from components.tranformers.modules.embedders.timestep_embedder import TimestepEmbedding
from components.tranformers.modules.timesteps import Timesteps


class CombinedTimestepLabelEmbeddings(Module):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(channel_count=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(input_channel_count=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        class_labels = self.class_embedder(class_labels)
        conditioning = timesteps_emb + class_labels

        return conditioning