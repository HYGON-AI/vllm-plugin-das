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

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.utils"
PATCH_ID = "worker.framework_opt.step5_token_fp8"
TARGETS = ("vllm.model_executor.layers.fused_moe.utils._fp8_quantize",)
_MARKER = "_vllm_hcu_step5_token_fp8_applied"
# Lazily cached ipc_fused_dispatch module; its UP02_DEFER_ALLOWED flag turns on the
# deferred-quant branch only where the fused all-gather is valid (world-2 DP decode).
_up02_state = None

SIGNATURES = (
    (
        "vllm.model_executor.layers.fused_moe.utils._fp8_quantize",
        ("A", "A_scale", "per_act_token", "block_shape"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os
    from vllm_hcu.platforms import envs as henvs

    original = require_callable(target, "_fp8_quantize", TARGETS[0])
    # UP-02: opt-in only.  When enabled, the per-token FP8 quant is deferred to the DP IPC
    # all-gather prologue (see ipc_fused_dispatch.py); the communicator falls back to a local
    # quant whenever the fused shape/capacity contract is not met (e.g. prefill).
    UP02 = os.environ.get("VLLM_HCU_UP02_FUSED_DISPATCH", "0") == "1"

    @functools.wraps(original)
    def quantize(A, A_scale, per_act_token, block_shape=None):
        if (
            henvs.optional_custom_op_enabled(
                os.environ.get("VLLM_HCU_TRITON_TOKEN_FP8_QUANT", "0") == "1"
            )
            and block_shape is None
            and per_act_token
            and A_scale is None
        ):
            from vllm_hcu.ops import token_fp8_quant as tq

            if tq.can_use(A):
                if UP02:
                    global _up02_state
                    if _up02_state is None:
                        from vllm_hcu.distributed.device_communicators import (
                            ipc_fused_dispatch as _up02_state,
                        )
                if (
                    UP02
                    and _up02_state is not None
                    and _up02_state.UP02_DEFER_ALLOWED
                    and A.dim() == 2
                    and A.shape[-1] % 4 == 0
                    and A.shape[0] > 0
                ):
                    # Defer the quant to the DP IPC all-gather prologue.  The placeholder
                    # keeps prepare()'s extra_tensors/scale path intact; the communicator
                    # replaces it with the gathered scale (or falls back to a local quant).
                    import torch

                    if not A.is_contiguous():
                        A = A.contiguous()
                    dummy = torch.empty((A.shape[0], 1), dtype=torch.float32, device=A.device)
                    A._hcu_up02_quant_pending = True
                    return A, dummy
                return tq.token_fp8_quant(A)
        return original(A, A_scale, per_act_token, block_shape)

    target._fp8_quantize = quantize
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
