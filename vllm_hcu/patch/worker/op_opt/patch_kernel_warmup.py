# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skip the NVIDIA-only MiniMax warmup import on HCU."""

from __future__ import annotations

import functools
import sys
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.warmup.kernel_warmup"
PATCH_ID = "worker.op_opt.warmup.skip_nvidia_minimax"
TARGETS = (f"{TARGET_MODULE}.kernel_warmup",)
_MINIMAX_WARMUP_MODULE = (
    "vllm.model_executor.warmup.minimax_m3_msa_warmup"
)
_MODULE_MARKER = "_vllm_hcu_kernel_warmup_applied"
_WRAPPER_MARKER = "_vllm_hcu_kernel_warmup_wrapper"
_MISSING = object()


def _noop_minimax_m3_msa_warmup(worker) -> None:
    del worker


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        target,
        _MODULE_MARKER,
        ((target, "kernel_warmup", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(target, "kernel_warmup", TARGETS[0])
    require_exact_signature(original, TARGETS[0], positional=("worker",))

    @functools.wraps(original)
    def hcu_kernel_warmup(worker):
        if not target.current_platform.is_rocm():
            return original(worker)

        previous = sys.modules.get(_MINIMAX_WARMUP_MODULE, _MISSING)
        stub = ModuleType(_MINIMAX_WARMUP_MODULE)
        stub.minimax_m3_msa_warmup = _noop_minimax_m3_msa_warmup
        sys.modules[_MINIMAX_WARMUP_MODULE] = stub
        try:
            return original(worker)
        finally:
            if previous is _MISSING:
                sys.modules.pop(_MINIMAX_WARMUP_MODULE, None)
            else:
                sys.modules[_MINIMAX_WARMUP_MODULE] = previous

    setattr(hcu_kernel_warmup, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_kernel_warmup = original
    target.kernel_warmup = hcu_kernel_warmup
    setattr(target, _MODULE_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
