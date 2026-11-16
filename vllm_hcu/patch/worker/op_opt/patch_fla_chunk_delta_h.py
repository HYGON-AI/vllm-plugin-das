# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route vLLM FLA chunk-state through the selected HCU GDN prefill provider."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._aiter_fla import make_aiter_gdn_resolver
from ._boltops_fla import make_boltops_gdn_resolver
from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._gdn_backend import (
    AITER,
    BOLTOPS,
    aiter_input_fallback,
    make_provider_selector,
)

TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.chunk"
PATCH_ID = "worker.op_opt.fla.chunk_delta_h.boltops"
TARGETS = (f"{TARGET_MODULE}.chunk_gated_delta_rule_fwd_h",)
_MARKER = "_vllm_hcu_fla_chunk_delta_h_applied"
_WRAPPER = "_vllm_hcu_fla_chunk_delta_h_wrapper"
_OP = "chunk_gated_delta_rule_fwd_h"


def _use_boltops(k, u) -> bool:
    """Keep the prior provider except for the audited official-faster shape."""
    return not (
        k.ndim == 4
        and u.ndim == 4
        and k.dtype == u.dtype == torch.bfloat16
        and k.shape[-2:] == (4, 128)
        and u.shape[-2:] == (12, 128)
    )


def _aiter_unsupported(k, u, g, gk, initial_state, chunk_size) -> str | None:
    """Return why the aiter HIP kernel rejects these inputs, or None."""
    if k.shape[-1] != 128 or u.shape[-1] != 128:
        return "head dims other than K=V=128"
    if chunk_size != 64:
        return "chunk_size other than 64"
    if k.dtype not in (torch.float16, torch.bfloat16):
        return f"{k.dtype} activations"
    if any(x is not None and x.dtype != torch.float32 for x in (g, gk)):
        return "non-fp32 gates"
    if initial_state is not None and initial_state.dtype != torch.float32:
        return f"{initial_state.dtype} initial_state"
    return None


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
    resolve_boltops = make_boltops_gdn_resolver(_OP)
    select_provider = make_provider_selector(
        "prefill",
        _OP,
        {BOLTOPS: resolve_boltops, AITER: make_aiter_gdn_resolver(_OP)},
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
        provider, kernel = select_provider()
        if provider == AITER:
            reason = _aiter_unsupported(k, u, g, gk, initial_state, chunk_size)
            if reason is not None:
                provider, kernel = aiter_input_fallback(
                    "prefill", _OP, reason, resolve_boltops
                )
        if provider == AITER:
            # Reuse the framework's precomputed varlen metadata; the HIP
            # kernel requires int64 chunk_offsets.
            if chunk_offsets is not None and chunk_offsets.dtype != torch.int64:
                chunk_offsets = chunk_offsets.to(torch.int64)
            return kernel(
                k, w, u, g, gk, initial_state, None, output_final_state,
                chunk_size, save_new_value, cu_seqlens, chunk_indices,
                chunk_offsets, use_exp2, True,
            )
        if provider != BOLTOPS or not _use_boltops(k, u):
            return original(
                k, w, u, g, gk, initial_state, output_final_state,
                chunk_size, save_new_value, cu_seqlens, chunk_indices,
                chunk_offsets, use_exp2,
            )

        # BoltOPs derives chunk offsets from cu_seqlens and chunk_size.
        return kernel(
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
            # vLLM reserves block/state index 0 as NULL_BLOCK_ID.
            null_state_index=0,
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
