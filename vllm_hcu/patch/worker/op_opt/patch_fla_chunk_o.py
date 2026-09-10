# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the vLLM FLA chunk-output kernel through BoltOPs on HCU."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.third_party.flash_linear_attention.ops.chunk"
PATCH_ID = "worker.op_opt.fla.chunk_o.boltops"
TARGETS = (f"{TARGET_MODULE}.chunk_fwd_o",)
_MARKER = "_vllm_hcu_fla_chunk_o_applied"
_WRAPPER = "_vllm_hcu_fla_chunk_o_wrapper"


def _enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA
    )


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    wrapped = ((chunk, "chunk_fwd_o", TARGETS[0], _WRAPPER),)
    if already_applied(chunk, _MARKER, wrapped):
        return False
    original = require_callable(chunk, "chunk_fwd_o", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "q", "k", "v", "h", "g", "scale", "cu_seqlens",
            "chunk_indices", "chunk_size", "core_attn_out",
        ),
        defaults={
            "g": None,
            "scale": None,
            "cu_seqlens": None,
            "chunk_indices": None,
            "chunk_size": chunk.FLA_CHUNK_SIZE,
            "core_attn_out": None,
        },
    )

    @functools.wraps(original)
    def hcu_chunk_o(
        q,
        k,
        v,
        h,
        g=None,
        scale=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk.FLA_CHUNK_SIZE,
        core_attn_out=None,
    ):
        if not _enabled():
            return original(
                q, k, v, h, g, scale, cu_seqlens, chunk_indices,
                chunk_size, core_attn_out,
            )
        try:
            from boltops.fla.gdn import chunk_fwd_o as boltops_kernel
        except ImportError:
            return original(
                q, k, v, h, g, scale, cu_seqlens, chunk_indices,
                chunk_size, core_attn_out,
            )

        boltops_output = boltops_kernel(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            g_gamma=None,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            use_exp2=False,
            transpose_state_layout=True,
            kernel_cfg=None,
        )
        if core_attn_out is None:
            return boltops_output
        if core_attn_out.numel() < v.numel():
            raise ValueError("core_attn_out is too small for HCU FLA chunk_o")
        out = core_attn_out[:v.numel()].view(*v.shape)
        out.copy_(boltops_output)
        return out

    setattr(hcu_chunk_o, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_chunk_fwd_o", original)
    setattr(chunk, "chunk_fwd_o", hcu_chunk_o)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
