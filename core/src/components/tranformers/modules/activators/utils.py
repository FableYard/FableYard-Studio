# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

from torch.nn import Module, SiLU, Mish, GELU, ReLU

ACT2CLS = {
    "swish": SiLU,
    "silu": SiLU,
    "mish": Mish,
    "gelu": GELU,
    "relu": ReLU,
}

def get_activation(activation_function: str) -> Module:
    """Helper function to get activation function from string.

    Args:
        activation_function (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    activation_function = activation_function.lower()
    if activation_function in ACT2CLS:
        return ACT2CLS[activation_function]()
    else:
        raise ValueError(
            f"activation function {activation_function} not found in ACT2FN mapping {list(ACT2CLS.keys())}"
        )
