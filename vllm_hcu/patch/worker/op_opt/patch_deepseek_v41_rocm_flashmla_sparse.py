# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Replace only V4.1's module-local sparse prefill call with default-on FlashMLA.

The shared ROCm op, V4.0, metadata builders, and decode calls stay untouched.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_exact_signature,
    require_unpatched,
)
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla import try_flashmla_prefill

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
PATCH_ID = "worker.op_opt.deepseek_v41.flashmla_sparse_prefill"
TARGETS = (f"{TARGET_MODULE}.rocm_sparse_attn_prefill",)
_MARKER = "_vllm_hcu_dsv41_flashmla_prefill_applied"
_WRAPPER = "_vllm_hcu_dsv41_flashmla_prefill_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    wrapped = ((target, "rocm_sparse_attn_prefill", TARGETS[0], _WRAPPER),)
    if already_applied(target, _MARKER, wrapped):
        return False
    original = require_unpatched(
        target, "rocm_sparse_attn_prefill", TARGETS[0], _WRAPPER
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "q",
            "kv",
            "indices",
            "topk_length",
            "scale",
            "head_dim",
            "nope_head_dim",
            "rope_head_dim",
            "attn_sink",
            "output",
            "ragged_indices",
            "ragged_indptr",
        ),
        defaults={"ragged_indices": None, "ragged_indptr": None},
    )

    @functools.wraps(original)
    def sparse_prefill(
        q,
        kv,
        indices,
        topk_length,
        scale,
        head_dim,
        nope_head_dim,
        rope_head_dim,
        attn_sink,
        output,
        ragged_indices=None,
        ragged_indptr=None,
    ):
        arguments = dict(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=topk_length,
            scale=scale,
            head_dim=head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            attn_sink=attn_sink,
            output=output,
            ragged_indices=ragged_indices,
            ragged_indptr=ragged_indptr,
        )
        selected = try_flashmla_prefill(**arguments)
        if selected:
            return None
        return original(**arguments)

    setattr(sparse_prefill, _WRAPPER, True)
    target._vllm_hcu_original_rocm_sparse_attn_prefill = original
    target.rocm_sparse_attn_prefill = sparse_prefill
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
