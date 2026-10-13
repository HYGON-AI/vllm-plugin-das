# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.v1.worker.gpu.dp_utils"
PATCH_ID = "worker.framework_opt.step5_dp_sync"
TARGETS = ("vllm.v1.worker.gpu.dp_utils.dispatch_cg_and_sync_dp",)
_MARKER = "_vllm_hcu_step5_dp_sync_applied"

SIGNATURES = (
    (
        "vllm.v1.worker.gpu.dp_utils.dispatch_cg_and_sync_dp",
        (
            "cudagraph_manager",
            "num_reqs",
            "num_tokens",
            "uniform_token_count",
            "dp_size",
            "dp_rank",
            "max_query_len",
            "need_eager",
            "num_active_loras",
            "parallel_config",
            "allow_ubatching",
            "uniform_decode",
            "dp_sync",
        ),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools

    original = require_callable(target, "dispatch_cg_and_sync_dp", TARGETS[0])

    @functools.wraps(original)
    def dispatch(
        cudagraph_manager,
        num_reqs,
        num_tokens,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=None,
        need_eager=False,
        num_active_loras=0,
        parallel_config=None,
        allow_ubatching=False,
        uniform_decode=False,
        dp_sync=None,
    ):
        if dp_sync is not None and dp_size > 1:
            token_mismatch = dp_sync.num_tokens_across_dp[dp_rank] != num_tokens
            uniform_mismatch = (
                dp_sync.uniform_token_count is not None
                and uniform_token_count is not None
                and uniform_token_count != dp_sync.uniform_token_count
            )
            if token_mismatch or uniform_mismatch:
                dp_sync = None
        return original(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len,
            need_eager,
            num_active_loras,
            parallel_config,
            allow_ubatching,
            uniform_decode,
            dp_sync,
        )

    target.dispatch_cg_and_sync_dp = dispatch
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
