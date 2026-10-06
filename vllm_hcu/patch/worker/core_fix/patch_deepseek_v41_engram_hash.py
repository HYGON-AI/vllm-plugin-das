# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use a correct Engram hash launch geometry on HCU."""

from __future__ import annotations

import inspect
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_callable

TARGET_MODULE = "vllm.models.deepseek_v41.common.engram"
PATCH_ID = "worker.core_fix.deepseek_v41.engram_hash_launch"
TARGETS = (f"{TARGET_MODULE}._hash_ids_kernel",)
_MARKER = "_vllm_hcu_dsv41_hash_launch_applied"


class _HashKernelLaunch:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def hcu_launch(*args, **kwargs):
            # gfx938 / Triton 3.6 drops 2-gram stores with BLOCK_T=32, four warps.
            kwargs["num_warps"] = 2
            return launch(*args, **kwargs)

        return hcu_launch

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def apply_to_module(module: ModuleType) -> bool:
    engram = load_exact_module(TARGET_MODULE, module)
    kernel = getattr(engram, "_hash_ids_kernel", None)
    if getattr(engram, _MARKER, False):
        if not isinstance(kernel, _HashKernelLaunch):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {PATCH_ID} is stale"
            )
        return False
    require_callable(kernel, "__getitem__", TARGETS[0])
    function = require_callable(kernel, "fn", TARGETS[0] + ".fn")
    required = {"output", "num_heads", "MAX_NGRAM", "BLOCK_T", "BLOCK_H"}
    if not required.issubset(inspect.signature(function).parameters):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible signature"
        )
    engram._vllm_hcu_original_hash_ids_kernel = kernel
    engram._hash_ids_kernel = _HashKernelLaunch(kernel)
    setattr(engram, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
