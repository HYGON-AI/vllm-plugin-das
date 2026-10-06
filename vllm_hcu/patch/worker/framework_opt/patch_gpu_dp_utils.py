# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep fixed DeepEP-LL CUDA-graph dispatch rank-local on MRV2."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)


TARGET_MODULE = "vllm.v1.worker.gpu.dp_utils"
PATCH_ID = "worker.framework_opt.dp.gpu_deepep_low_latency"
TARGETS = (f"{TARGET_MODULE}.sync_cudagraph_and_dp_padding",)
_MARKER = "_vllm_hcu_gpu_dp_low_latency_applied"
_WRAPPER = "_vllm_hcu_gpu_dp_low_latency_wrapper"


def _uses_fixed_deepep_low_latency(parallel_config: object | None) -> bool:
    return bool(
        parallel_config is not None
        and getattr(parallel_config, "all2all_backend", None)
        == "deepep_low_latency"
        and not getattr(parallel_config, "_vllm_hcu_deepep_auto", False)
    )


def apply_to_module(module: ModuleType) -> bool:
    dp = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (
            dp,
            "sync_cudagraph_and_dp_padding",
            TARGETS[0],
            _WRAPPER,
        ),
    )
    if already_applied(dp, _MARKER, wrapped):
        return False
    original = require_callable(
        dp,
        "sync_cudagraph_and_dp_padding",
        TARGETS[0],
    )
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
            "max_query_len",
            "num_active_loras",
            "parallel_config",
            "allow_ubatching",
            "uniform_decode",
        ),
        defaults={
            "max_query_len": None,
            "num_active_loras": 0,
            "parallel_config": None,
            "allow_ubatching": False,
            "uniform_decode": False,
        },
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
        max_query_len=None,
        num_active_loras=0,
        parallel_config=None,
        allow_ubatching=False,
        uniform_decode=False,
    ):
        if _uses_fixed_deepep_low_latency(parallel_config):
            return desired_batch_desc, None
        return original(
            cudagraph_manager,
            desired_batch_desc,
            num_tokens,
            num_reqs,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
            parallel_config=parallel_config,
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode,
        )

    setattr(hcu_sync, _WRAPPER, True)
    setattr(
        dp,
        "_vllm_hcu_original_sync_cudagraph_and_dp_padding",
        original,
    )
    setattr(dp, "sync_cudagraph_and_dp_padding", hcu_sync)
    setattr(dp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
