# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route Qwen4Exp QSA wrappers through the QSA-specific kernel dispatcher.

The official QSA module owns metadata construction, cache views, top-k
selection, and index expansion.  This adapter only replaces the two audited
compute wrappers when the QSA dispatcher selects HCU CUTLASS mode; the
official Triton implementations remain the fallback for every other
attention mode.  QSA is deliberately not added to vLLM's global
``AttentionBackendEnum``.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from vllm_hcu.v1.attention.backends.qsa import (
    QSA_BACKEND_FLASH,
    get_qsa_flash_attn_mode,
    get_qsa_kernel_backend,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ops.qsa"
PATCH_ID = "worker.op_opt.qwen4_exp.qsa.flash_attn"
TARGETS = (
    f"{TARGET_MODULE}.qsa_mqa_paged",
    f"{TARGET_MODULE}.qsa_sparse_paged_attention",
)
_MARKER = "_vllm_hcu_qwen4_exp_qsa_flash_attn_applied"
_WRAPPER = "_vllm_hcu_qsa_flash_attn_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    qsa = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (qsa, "qsa_mqa_paged", TARGETS[0], _WRAPPER),
        (qsa, "qsa_sparse_paged_attention", TARGETS[1], _WRAPPER),
    )
    if already_applied(qsa, _MARKER, wrapped):
        return False

    original_mqa = require_callable(qsa, "qsa_mqa_paged", TARGETS[0])
    require_exact_signature(
        original_mqa,
        TARGETS[0],
        positional=(
            "q",
            "k_cache",
            "page_table",
            "token_to_req",
            "query_positions",
            "sequence_lengths",
            "compress_ratio",
            "num_columns",
            "score_scale",
        ),
        defaults={"num_columns": None, "score_scale": None},
    )
    original_sparse = require_callable(
        qsa, "qsa_sparse_paged_attention", TARGETS[1]
    )
    require_exact_signature(
        original_sparse,
        TARGETS[1],
        positional=(
            "q",
            "k_cache",
            "v_cache",
            "logical_indices",
            "block_table",
            "token_to_req",
            "out",
        ),
        defaults={"out": None},
    )

    # Temporary one-shot diagnostics for the first real QSA invocation in
    # each worker.  Keeping the flags local avoids printing once per token.
    mqa_route_logged = False
    sparse_route_logged = False

    @functools.wraps(original_mqa)
    def hcu_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        nonlocal mqa_route_logged
        mode = get_qsa_flash_attn_mode()
        try:
            backend = get_qsa_kernel_backend(
                mode,
                triton_mqa_paged=original_mqa,
                triton_sparse_gqa_paged_attn=original_sparse,
            )
        except RuntimeError as exc:
            raise PatchCompatibilityError(str(exc)) from exc
        if not mqa_route_logged:
            route = (
                "flash_attn.mqa_paged_score_func"
                if backend.name == QSA_BACKEND_FLASH
                else "vllm QSA Triton kernel"
            )
            print(
                f"[HCU QSA][{mode.upper()}] using {route}",
                flush=True,
            )
            mqa_route_logged = True
        return backend.mqa_paged_score(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
            num_columns=num_columns,
            score_scale=score_scale,
        )

    @functools.wraps(original_sparse)
    def hcu_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
    ):
        nonlocal sparse_route_logged
        mode = get_qsa_flash_attn_mode()
        try:
            backend = get_qsa_kernel_backend(
                mode,
                triton_mqa_paged=original_mqa,
                triton_sparse_gqa_paged_attn=original_sparse,
            )
        except RuntimeError as exc:
            raise PatchCompatibilityError(str(exc)) from exc
        if not sparse_route_logged:
            route = (
                "flash_attn.sparse_gqa_paged_attn_func"
                if backend.name == QSA_BACKEND_FLASH
                else "vllm QSA Triton kernel"
            )
            print(
                f"[HCU QSA][{mode.upper()}] using {route}",
                flush=True,
            )
            sparse_route_logged = True
        return backend.sparse_gqa_paged_attn(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            out=out,
        )

    setattr(hcu_mqa_paged, _WRAPPER, True)
    setattr(hcu_sparse_paged_attention, _WRAPPER, True)
    setattr(qsa, "_vllm_hcu_original_qsa_mqa_paged", original_mqa)
    setattr(
        qsa,
        "_vllm_hcu_original_qsa_sparse_paged_attention",
        original_sparse,
    )
    setattr(qsa, "qsa_mqa_paged", hcu_mqa_paged)
    setattr(qsa, "qsa_sparse_paged_attention", hcu_sparse_paged_attention)
    setattr(qsa, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
