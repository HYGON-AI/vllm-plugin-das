# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Repair inconsistent Qwen3.5 MTP shared-gate quantization metadata."""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Iterator
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_5_mtp"
PATCH_ID = "worker.core_fix.qwen3_5_mtp.dequantize_ignored_shared_gate"
TARGET_SYMBOL = f"{TARGET_MODULE}.Qwen3_5MTP.load_weights"
_CLASS_MARKER = "_vllm_hcu_qwen3_5_mtp_shared_gate_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen3_5_mtp_shared_gate_wrapper"
_SERIALIZED_GATE = re.compile(
    r"^mtp\.layers\.(?P<layer>\d+)\.mlp\.shared_expert_gate\."
    r"(?P<kind>weight|weight_scale)$"
)


def _repair_ignored_shared_gate_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    parameters: dict[str, torch.nn.Parameter | torch.Tensor],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dequantize only an INT8+scale pair whose runtime layer is unquantized."""

    pending: dict[int, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, tensor in weights:
        match = _SERIALIZED_GATE.fullmatch(name)
        if match is None:
            yield name, tensor
            continue

        layer = int(match.group("layer"))
        runtime_base = f"model.layers.{layer}.mlp.shared_expert_gate"
        if f"{runtime_base}.weight_scale" in parameters:
            yield name, tensor
            continue

        pair = pending.setdefault(layer, {})
        pair[match.group("kind")] = (name, tensor)
        if pair.keys() != {"weight", "weight_scale"}:
            continue

        weight_name, weight = pair["weight"]
        scale_name, scale = pair["weight_scale"]
        runtime_weight = parameters.get(f"{runtime_base}.weight")
        if runtime_weight is not None and weight.dtype is torch.int8:
            dtype = runtime_weight.dtype
            yield weight_name, weight.to(dtype=dtype) * scale.to(dtype=dtype)
        else:
            yield weight_name, weight
            yield scale_name, scale
        del pending[layer]

    for layer in sorted(pending):
        pair = pending[layer]
        for kind in ("weight", "weight_scale"):
            if kind in pair:
                yield pair[kind]


def apply_to_module(module: ModuleType) -> bool:
    qwen_mtp = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        qwen_mtp,
        "Qwen3_5MTP",
        f"{TARGET_MODULE}.Qwen3_5MTP",
    )
    original = require_callable(model_class, "load_weights", TARGET_SYMBOL)
    if getattr(model_class, _CLASS_MARKER, False):
        current = vars(model_class).get("load_weights")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=("self", "weights"),
    )

    @functools.wraps(original)
    def hcu_load_weights(self, weights):
        parameters = dict(self.named_parameters())
        repaired = _repair_ignored_shared_gate_weights(weights, parameters)
        return original(self, repaired)

    setattr(hcu_load_weights, _WRAPPER_MARKER, True)
    setattr(model_class, "_vllm_hcu_original_load_weights", original)
    setattr(model_class, "load_weights", hcu_load_weights)
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
