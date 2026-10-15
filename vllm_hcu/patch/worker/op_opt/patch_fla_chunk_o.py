# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the vLLM FLA chunk-output kernel through the selected HCU provider."""

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
PATCH_ID = "worker.op_opt.fla.chunk_o.boltops"
TARGETS = (f"{TARGET_MODULE}.chunk_fwd_o",)
_MARKER = "_vllm_hcu_fla_chunk_o_applied"
_WRAPPER = "_vllm_hcu_fla_chunk_o_wrapper"
_OP = "chunk_fwd_o"


def _output_buffer(v, core_attn_out):
    if core_attn_out is None:
        return torch.empty_like(v)
    if core_attn_out.numel() < v.numel():
        raise ValueError("core_attn_out is too small for HCU FLA chunk_o")
    return core_attn_out[:v.numel()].view(*v.shape)


def _aiter_unsupported(q, v, h, g, chunk_size) -> str | None:
    """Return why the aiter HIP kernel rejects these inputs, or None."""
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        return "head dims other than K=V=128"
    if chunk_size != 64:
        return "chunk_size other than 64"
    if q.dtype not in (torch.float16, torch.bfloat16) or h.dtype != q.dtype:
        return f"{q.dtype} activations with {h.dtype} state"
    if g is not None and g.dtype != torch.float32:
        return "non-fp32 gates"
    return None


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
    resolve_boltops = make_boltops_gdn_resolver(_OP)
    select_provider = make_provider_selector(
        "prefill",
        _OP,
        {BOLTOPS: resolve_boltops, AITER: make_aiter_gdn_resolver(_OP)},
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
        provider, kernel = select_provider()
        if provider == AITER:
            reason = _aiter_unsupported(q, v, h, g, chunk_size)
            if reason is not None:
                provider, kernel = aiter_input_fallback(
                    "prefill", _OP, reason, resolve_boltops
                )
        if provider not in (AITER, BOLTOPS):
            return original(
                q, k, v, h, g, scale, cu_seqlens, chunk_indices,
                chunk_size, core_attn_out,
            )

        # aiter HIP and BoltOPs share this contract and return a fresh output.
        output = kernel(
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
            return output
        out = _output_buffer(v, core_attn_out)
        out.copy_(output)
        return out

    setattr(hcu_chunk_o, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_chunk_fwd_o", original)
    setattr(chunk, "chunk_fwd_o", hcu_chunk_o)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
