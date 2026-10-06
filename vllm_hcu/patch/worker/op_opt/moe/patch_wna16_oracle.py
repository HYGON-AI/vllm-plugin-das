# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow explicit AITER selection for the HCU WNA16 runtime adapter."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"
PATCH_ID = "worker.op_opt.moe.oracle.wna16_aiter"
TARGETS = (
    f"{TARGET_MODULE}.WNA16MoEBackend",
    f"{TARGET_MODULE}.map_wna16_backend",
)
_MARKER = "_vllm_hcu_wna16_aiter_oracle_applied"
_WRAPPER_MARKER = "_vllm_hcu_wna16_aiter_mapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        mapper = require_callable(target, "map_wna16_backend", TARGETS[1])
        if not getattr(mapper, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"stale HCU WNA16 oracle marker for {TARGET_MODULE}; restart process"
            )
        return False

    backend_enum = require_class(target, "WNA16MoEBackend", TARGETS[0])
    triton_backend = getattr(backend_enum, "TRITON", None)
    if triton_backend is None:
        raise PatchCompatibilityError(f"{TARGETS[0]} has no TRITON member")
    original = require_callable(target, "map_wna16_backend", TARGETS[1])
    require_parameter_names(original, TARGETS[1], ("runner_backend",))

    @functools.wraps(original)
    def hcu_map_wna16_backend(runner_backend):
        # vLLM has no AITER member in its WNA16 oracle.  The HCU compressed-
        # tensors adapter dispatches the actual W4A16 call to AITER and needs
        # the official Triton experts object only as its validated fallback.
        if runner_backend == "aiter":
            return triton_backend
        return original(runner_backend)

    setattr(hcu_map_wna16_backend, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_map_wna16_backend = original
    target.map_wna16_backend = hcu_map_wna16_backend
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
