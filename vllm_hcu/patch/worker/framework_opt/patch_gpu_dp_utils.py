# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skip MRV2 cross-DP CG padding gloo allreduce for DeepEP-LL."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.gpu.dp_utils"
PATCH_ID = "worker.framework_opt.dp.gpu_deepep_low_latency"
TARGETS = (f"{TARGET_MODULE}.sync_cudagraph_and_dp_padding",)
_MARKER = "_vllm_hcu_gpu_dp_low_latency_applied"
_WRAPPER = "_vllm_hcu_gpu_dp_low_latency_wrapper"

# Set by apply_worker_patches (explicit deepep_low_latency, not deepep_auto).
_SKIP = False


def bind_skip_cross_dp_cg_sync(*, enabled: bool) -> None:
    global _SKIP
    _SKIP = bool(enabled)


def apply_to_module(module: ModuleType) -> bool:
    dp = load_exact_module(TARGET_MODULE, module)
    wrapped = ((dp, "sync_cudagraph_and_dp_padding", TARGETS[0], _WRAPPER),)
    if already_applied(dp, _MARKER, wrapped):
        return False
    original = require_callable(dp, "sync_cudagraph_and_dp_padding", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "cudagraph_manager",
            "desired_batch_desc",
            "num_tokens",
            "num_reqs",
            "uniform_token_count",
            "dp_size",
            "dp_rank",
            "num_active_loras",
        ),
        defaults={"num_active_loras": 0},
    )

    @functools.wraps(original)
    def hcu_sync(
        cudagraph_manager,
        desired_batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        num_active_loras=0,
    ):
        if _SKIP:
            return desired_batch_desc, None
        return original(
            cudagraph_manager,
            desired_batch_desc,
            num_tokens,
            num_reqs,
            uniform_token_count,
            dp_size,
            dp_rank,
            num_active_loras=num_active_loras,
        )

    setattr(hcu_sync, _WRAPPER, True)
    setattr(dp, "_vllm_hcu_original_sync_cudagraph_and_dp_padding", original)
    setattr(dp, "sync_cudagraph_and_dp_padding", hcu_sync)
    setattr(dp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "bind_skip_cross_dp_cg_sync",
]
