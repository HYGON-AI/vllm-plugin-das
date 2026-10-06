# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fit MiniMax-M3's Triton sparse decode kernel in 64-KiB LDS."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.models.minimax_m3.common.ops.sparse_attn"
PATCH_ID = "worker.core_fix.minimax_m3.sparse_decode_one_stage"
_MARKER = "_vllm_hcu_minimax_m3_sparse_decode_one_stage_applied"
_PROXY_MARKER = "_vllm_hcu_sparse_decode_one_stage_proxy"


class _OneStageKernel:
    """Preserve Triton's ``kernel[grid](...)`` API and constrain staging."""

    def __init__(self, kernel: Any):
        self.kernel = kernel
        setattr(self, _PROXY_MARKER, True)

    def __getitem__(self, grid: Any):
        launch = self.kernel[grid]

        def launch_one_stage(*args: Any, **kwargs: Any):
            kwargs.setdefault("num_stages", 1)
            return launch(*args, **kwargs)

        return launch_one_stage


def apply_to_module(module: ModuleType) -> bool:
    sparse = load_exact_module(TARGET_MODULE, module)
    if getattr(sparse, _MARKER, False):
        if not getattr(
            getattr(sparse, "_gqa_sparse_decode_kernel", None),
            _PROXY_MARKER,
            False,
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    kernel = getattr(sparse, "_gqa_sparse_decode_kernel", None)
    if kernel is None or not callable(getattr(kernel, "__getitem__", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            "_gqa_sparse_decode_kernel is missing"
        )

    sparse._vllm_hcu_original_gqa_sparse_decode_kernel = kernel
    sparse._gqa_sparse_decode_kernel = _OneStageKernel(kernel)
    setattr(sparse, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
