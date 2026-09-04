# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Metadata/validation for the custom-op-bearing HCU MoE runner replacement."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from .._common import require_exact_signature
from ._common import (
    PatchCompatibilityError,
    require_callable,
    require_class,
    require_replacement_module,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
REPLACEMENT_MODULE = "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
PATCH_ID = "worker.op_opt.moe.runner"
TARGETS = (
    TARGET_MODULE,
    f"{TARGET_MODULE}._moe_forward",
    f"{TARGET_MODULE}._moe_forward_fake",
    f"{TARGET_MODULE}._moe_forward_shared",
    f"{TARGET_MODULE}._moe_forward_shared_fake",
    f"{TARGET_MODULE}._moe_forward_shared_inplace",
    f"{TARGET_MODULE}._moe_forward_shared_inplace_fake",
    f"{TARGET_MODULE}.MoERunner.apply_routed_input_transform",
    f"{TARGET_MODULE}.MoERunner._maybe_apply_shared_experts",
    f"{TARGET_MODULE}.MoERunner._quant_method_supports_quanted_inputs",
    f"{TARGET_MODULE}.MoERunner._apply_quant_method",
    f"{TARGET_MODULE}.MoERunner._maybe_sync_shared_experts_stream",
    f"{TARGET_MODULE}.MoERunner.forward",
    f"{TARGET_MODULE}.MoERunner._forward_impl",
)
_MARKER = "_vllm_hcu_moe_runner_replacement_validated"


def _names(function) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def apply_to_module(module: ModuleType) -> bool:
    require_replacement_module(module, REPLACEMENT_MODULE, TARGETS)
    if getattr(module, _MARKER, False):
        return False
    runner = require_class(module, "MoERunner", f"{TARGET_MODULE}.MoERunner")
    expected = {
        "_moe_forward": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
        "_moe_forward_fake": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
        "_moe_forward_shared": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
        "_moe_forward_shared_fake": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
        "_moe_forward_shared_inplace": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
        "_moe_forward_shared_inplace_fake": (
            "hidden_states", "router_logits", "shared_experts_input", "input_ids",
            "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
            "layer_name", "hidden_dim_unpadded",
        ),
    }
    for name, names in expected.items():
        function = require_callable(module, name, f"{TARGET_MODULE}.{name}")
        if _names(function) != names:
            raise PatchCompatibilityError(
                f"HCU replacement {REPLACEMENT_MODULE}.{name} has incompatible "
                f"signature {inspect.signature(function)}"
            )
    input_transform = require_callable(
        runner,
        "apply_routed_input_transform",
        TARGETS[7],
    )
    require_exact_signature(
        input_transform,
        TARGETS[7],
        positional=("self", "hidden_states"),
    )
    if _names(runner.forward) != (
        "self", "hidden_states", "router_logits", "input_ids",
        "quanted_hidden_states", "scale", "topk_weights", "topk_ids",
    ):
        raise PatchCompatibilityError("HCU MoERunner.forward schema is incompatible")
    setattr(module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    if module is None:
        module = importlib.import_module(REPLACEMENT_MODULE)
    return apply_to_module(module)


__all__ = [
    "PATCH_ID",
    "REPLACEMENT_MODULE",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
