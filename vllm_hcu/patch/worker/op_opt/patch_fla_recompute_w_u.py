# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the vLLM FLA recompute_w_u helper through the selected HCU provider.

aiter has no recompute_w_u kernel, so ``hcu-aiter`` falls back to BoltOPs.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._boltops_fla import make_boltops_gdn_resolver
from ._gdn_backend import BOLTOPS, make_provider_selector

TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.chunk"
PATCH_ID = "worker.op_opt.fla.recompute_w_u.boltops"
TARGETS = (f"{TARGET_MODULE}.recompute_w_u_fwd",)
_MARKER = "_vllm_hcu_fla_recompute_w_u_applied"
_WRAPPER = "_vllm_hcu_fla_recompute_w_u_wrapper"
_OP = "recompute_w_u_fwd"


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    wrapped = ((chunk, "recompute_w_u_fwd", TARGETS[0], _WRAPPER),)
    if already_applied(chunk, _MARKER, wrapped):
        return False
    original = require_callable(chunk, "recompute_w_u_fwd", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "k", "v", "beta", "g_cumsum", "A", "cu_seqlens",
            "chunk_indices",
        ),
        defaults={"chunk_indices": None},
    )
    select_provider = make_provider_selector(
        "prefill", _OP, {BOLTOPS: make_boltops_gdn_resolver(_OP)}
    )

    @functools.wraps(original)
    def hcu_recompute_w_u(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
        chunk_indices=None,
    ):
        provider, boltops_kernel = select_provider()
        if provider != BOLTOPS:
            return original(
                k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices,
            )
        return boltops_kernel(
            k,
            v,
            beta,
            g_cumsum,
            A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    setattr(hcu_recompute_w_u, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_recompute_w_u_fwd", original)
    setattr(chunk, "recompute_w_u_fwd", hcu_recompute_w_u)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
