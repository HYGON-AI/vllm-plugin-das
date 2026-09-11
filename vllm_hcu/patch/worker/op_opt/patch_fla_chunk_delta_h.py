# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route supported vLLM FLA chunk-state shapes through BoltOPs on HCU."""

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

TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.chunk"
PATCH_ID = "worker.op_opt.fla.chunk_delta_h.boltops"
TARGETS = (f"{TARGET_MODULE}.chunk_gated_delta_rule_fwd_h",)
_MARKER = "_vllm_hcu_fla_chunk_delta_h_applied"
_WRAPPER = "_vllm_hcu_fla_chunk_delta_h_wrapper"


def _enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA
    )


def _use_boltops(k, u) -> bool:
    """Use BoltOPs only for head ratios where it beats current vLLM FLA."""
    return (
        k.ndim == 4
        and u.ndim == 4
        and u.shape[-2] <= 2 * k.shape[-2]
    )


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    wrapped = ((chunk, "chunk_gated_delta_rule_fwd_h", TARGETS[0], _WRAPPER),)
    if already_applied(chunk, _MARKER, wrapped):
        return False
    original = require_callable(chunk, "chunk_gated_delta_rule_fwd_h", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "k", "w", "u", "g", "gk", "initial_state", "output_final_state",
            "chunk_size", "save_new_value", "cu_seqlens", "chunk_indices",
            "chunk_offsets", "use_exp2",
        ),
        defaults={
            "g": None,
            "gk": None,
            "initial_state": None,
            "output_final_state": False,
            "chunk_size": chunk.FLA_CHUNK_SIZE,
            "save_new_value": True,
            "cu_seqlens": None,
            "chunk_indices": None,
            "chunk_offsets": None,
            "use_exp2": False,
        },
    )
    resolve_boltops = make_boltops_gdn_resolver(
        "chunk_gated_delta_rule_fwd_h"
    )

    @functools.wraps(original)
    def hcu_chunk_delta_h(
        k,
        w,
        u,
        g=None,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk.FLA_CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
        use_exp2=False,
    ):
        if not _enabled() or not _use_boltops(k, u):
            return original(
                k, w, u, g, gk, initial_state, output_final_state,
                chunk_size, save_new_value, cu_seqlens, chunk_indices,
                chunk_offsets, use_exp2,
            )
        boltops_kernel = resolve_boltops()
        if boltops_kernel is None:
            return original(
                k, w, u, g, gk, initial_state, output_final_state,
                chunk_size, save_new_value, cu_seqlens, chunk_indices,
                chunk_offsets, use_exp2,
            )

        # BoltOPs derives chunk offsets from cu_seqlens and chunk_size.
        return boltops_kernel(
            k,
            w,
            u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=None,
            output_final_state=output_final_state,
            inplace_final_state=False,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            transpose_state_layout=True,
            kernel_cfg=None,
        )

    setattr(hcu_chunk_delta_h, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_chunk_gated_delta_rule_fwd_h", original)
    setattr(chunk, "chunk_gated_delta_rule_fwd_h", hcu_chunk_delta_h)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
