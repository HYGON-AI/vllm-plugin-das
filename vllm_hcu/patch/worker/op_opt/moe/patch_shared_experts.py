# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Metadata/validation for the HCU shared-experts module replacement."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from .._common import require_exact_signature
from ._common import PatchCompatibilityError, require_class, require_replacement_module

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.runner.shared_experts"
REPLACEMENT_MODULE = "vllm_hcu.model_executor.layers.fused_moe.shared_experts"
PATCH_ID = "worker.op_opt.moe.runner.shared_experts"
TARGETS = (
    TARGET_MODULE,
    f"{TARGET_MODULE}.SharedExperts._layer_supports_x_and_scale_quanted",
    f"{TARGET_MODULE}.SharedExperts._run_layer",
    f"{TARGET_MODULE}.SharedExperts._disable_shared_experts_overlap",
    f"{TARGET_MODULE}.SharedExperts._determine_shared_experts_order",
    f"{TARGET_MODULE}.SharedExperts.requires_input_preservation",
    f"{TARGET_MODULE}.SharedExperts.allows_inplace_routed_output",
    f"{TARGET_MODULE}.SharedExperts._should_run_shared_in_aux_stream",
    f"{TARGET_MODULE}.SharedExperts.maybe_sync_shared_experts_stream",
    f"{TARGET_MODULE}.SharedExperts._launch_in_aux_stream",
    f"{TARGET_MODULE}.SharedExperts._run_in_aux_stream",
    f"{TARGET_MODULE}.SharedExperts.output",
    f"{TARGET_MODULE}.SharedExperts.forward",
)
_MARKER = "_vllm_hcu_shared_experts_replacement_validated"


def apply_to_module(module: ModuleType) -> bool:
    require_replacement_module(module, REPLACEMENT_MODULE, TARGETS)
    if getattr(module, _MARKER, False):
        return False
    cls = require_class(module, "SharedExperts", f"{TARGET_MODULE}.SharedExperts")
    require_exact_signature(
        getattr(cls, "requires_input_preservation", None),
        TARGETS[5],
        positional=("self", "hidden_states"),
    )
    require_exact_signature(
        getattr(cls, "allows_inplace_routed_output", None),
        f"{TARGET_MODULE}.SharedExperts.allows_inplace_routed_output",
        positional=("self", "routed_input", "shared_input"),
    )
    expected = {
        "maybe_sync_shared_experts_stream": ("self", "shared_experts_input", "x_and_scale_quanted"),
        "_run_in_aux_stream": ("self", "shared_experts_input", "x_and_scale_quanted"),
        "forward": (
            "self",
            "shared_experts_input",
            "order",
            "x_and_scale_quanted",
        ),
    }
    for name, names in expected.items():
        function = getattr(cls, name, None)
        if not callable(function) or tuple(inspect.signature(function).parameters) != names:
            raise PatchCompatibilityError(
                f"HCU shared-experts replacement has incompatible {name} signature"
            )
    output = vars(cls).get("output")
    if not isinstance(output, property) or output.fget is None:
        raise PatchCompatibilityError("HCU SharedExperts.output property is missing")
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
