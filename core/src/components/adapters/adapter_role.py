# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from enum import Enum


class AdapterRole(Enum):
    ATTENTION = "attention"
    ATTENTION_CONTEXT = "attention_context"
    FFN = "ffn"
    FFN_CONTEXT = "ffn_context"
    ADALN = "adaln"
    ADALN_CONTEXT = "adaln_context"
    NORM_DANGEROUS = "norm_dangerous"