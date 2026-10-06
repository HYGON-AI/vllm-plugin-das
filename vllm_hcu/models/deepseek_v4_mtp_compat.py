# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility adapter for native DeepSeek V4 MTP weight loading."""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.models.deepseek_v4.amd.mtp import (
    DeepSeekV4MTP as NativeDeepSeekV4MTP,
)


class DeepSeekV4MTP(NativeDeepSeekV4MTP):
    """Accept compressed-tensors scale names in the native V4 MTP loader.

    The native loader normalizes generic checkpoint ``.scale`` keys to
    ``.weight_scale_inv`` before looking them up. Compressed-tensors W8A8
    layers register the same parameter as ``.weight_scale`` instead. Expose
    an alias only while the native loader builds its parameter dictionary so
    inference-time parameter enumeration remains unchanged.
    """

    _hcu_expose_scale_aliases = False

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        parameters = list(
            super().named_parameters(
                prefix=prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            )
        )
        yield from parameters

        if not self._hcu_expose_scale_aliases:
            return

        existing = {name for name, _ in parameters}
        for name, parameter in parameters:
            alias = None
            if name.endswith(".weight_scale"):
                alias = name.removesuffix(".weight_scale") + ".weight_scale_inv"
            elif name.endswith("_weight_scale"):
                alias = name.removesuffix("_weight_scale") + "_weight_scale_inv"
            elif name.endswith(".weight_scale_inv"):
                alias = name.removesuffix(".weight_scale_inv") + ".weight_scale"
            elif name.endswith("_weight_scale_inv"):
                alias = name.removesuffix("_weight_scale_inv") + "_weight_scale"

            if alias is not None and alias not in existing:
                yield alias, parameter

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        previous = self._hcu_expose_scale_aliases
        self._hcu_expose_scale_aliases = True
        try:
            return super().load_weights(weights)
        finally:
            self._hcu_expose_scale_aliases = previous
