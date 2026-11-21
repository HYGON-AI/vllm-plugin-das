# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Pass ``is_quantization`` only to loaders that accept it.

HCU column and row linear layers pass ``is_quantization`` into
``load_column_parallel_weight`` / ``load_row_parallel_weight``. Official
parameter classes accept it after the NN-layout patch. The GGUF plugin's
mixin still takes only ``loaded_weight``. The check is on the HCU call site,
so a spawned worker picks it up by importing the exchanged linear module.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def accepts_is_quantization(loader: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return False
    return "is_quantization" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def call_parallel_weight_loader(
    loader: Callable[..., Any],
    loaded_weight: Any,
    *,
    is_quantization: bool,
) -> Any:
    """Call a column or row loader, omitting the HCU flag when unsupported."""

    if accepts_is_quantization(loader):
        return loader(loaded_weight=loaded_weight, is_quantization=is_quantization)
    return loader(loaded_weight=loaded_weight)


__all__ = [
    "accepts_is_quantization",
    "call_parallel_weight_loader",
]
