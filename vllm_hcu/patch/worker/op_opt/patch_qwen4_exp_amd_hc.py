# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validation for the HCU Qwen4Exp AMD HyperConnection replacement."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from .moe._common import (
    PatchCompatibilityError,
    require_callable,
    require_replacement_module,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ops.hc"
REPLACEMENT_MODULE = "vllm_hcu.runtime_compat.qwen4_exp_amd_hc"
PATCH_ID = "worker.op_opt.qwen4_exp.amd.ops.hc"
TARGETS = (
    TARGET_MODULE,
    f"{TARGET_MODULE}._hc_gate_mix",
    f"{TARGET_MODULE}._hc_combine",
    f"{TARGET_MODULE}._hc_combine_norm",
)
_MARKER = "_vllm_hcu_qwen4_exp_hc_replacement_validated"


def _parameter_names(function) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def apply_to_module(module: ModuleType) -> bool:
    require_replacement_module(module, REPLACEMENT_MODULE, TARGETS)
    if getattr(module, _MARKER, False):
        return False

    expected = {
        "_hc_gate_mix": ("x", "gate", "hc_count"),
        "_hc_combine": (
            "residual",
            "block_output",
            "injection_logits",
            "hc_count",
        ),
        "_hc_combine_norm": (
            "residual",
            "block_output",
            "injection_logits",
            "norm_weight",
            "eps",
            "hc_count",
        ),
    }
    for name, names in expected.items():
        function = require_callable(module, name, f"{REPLACEMENT_MODULE}.{name}")
        if _parameter_names(function) != names:
            raise PatchCompatibilityError(
                f"HCU Qwen4Exp HC replacement has incompatible {name} "
                f"signature {inspect.signature(function)}"
            )

    for name, names in {
        "grouped_gemma_rmsnorm": ("x", "weight", "eps", "num_groups"),
        "hc_silu": ("x", "hc_count"),
        "hc_gate_mix": ("x", "gate", "hc_count"),
        "hc_combine": (
            "residual",
            "block_output",
            "injection_logits",
            "hc_count",
        ),
        "hc_combine_norm": (
            "residual",
            "block_output",
            "injection_logits",
            "norm_weight",
            "eps",
            "hc_count",
        ),
    }.items():
        function = require_callable(module, name, f"{REPLACEMENT_MODULE}.{name}")
        if _parameter_names(function) != names:
            raise PatchCompatibilityError(
                f"HCU Qwen4Exp HC replacement has incompatible {name} "
                f"signature {inspect.signature(function)}"
            )

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
