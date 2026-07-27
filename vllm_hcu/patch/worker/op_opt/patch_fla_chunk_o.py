# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the HCU AITER FLA output kernel without source mutation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.model_executor.layers.fla.ops.chunk_o"
PATCH_ID = "worker.op_opt.fla.chunk_o.aiter"
TARGETS = (f"{TARGET_MODULE}.chunk_fwd_o",)
_MARKER = "_vllm_hcu_chunk_o_applied"
_WRAPPER = "_vllm_hcu_chunk_o_wrapper"


def _enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA and henvs.VLLM_HCU_USE_CUSTOM_OPS)


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    if already_applied(chunk, _MARKER, ((chunk, "chunk_fwd_o", TARGETS[0], _WRAPPER),)):
        return False
    original = require_callable(chunk, "chunk_fwd_o", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("q", "k", "v", "h", "g", "scale", "cu_seqlens", "chunk_indices", "chunk_size", "core_attn_out"),
        defaults={"g": None, "scale": None, "cu_seqlens": None, "chunk_indices": None,
                  "chunk_size": chunk.FLA_CHUNK_SIZE, "core_attn_out": None},
    )

    @functools.wraps(original)
    def hcu_chunk_o(q, k, v, h, g=None, scale=None, cu_seqlens=None,
                    chunk_indices=None, chunk_size=chunk.FLA_CHUNK_SIZE,
                    core_attn_out=None):
        if not _enabled():
            return original(q, k, v, h, g, scale, cu_seqlens, chunk_indices,
                            chunk_size, core_attn_out)
        try:
            from aiter.ops.triton.fla.vllm.chunk_o import launch_chunk_fwd_kernel_o
        except ImportError as exc:
            raise RuntimeError("HCU AITER FLA chunk_o is enabled but unavailable") from exc
        B, T, Hg, K, V = *q.shape, v.shape[-1]
        H, BT = v.shape[-2], chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = chunk.prepare_chunk_indices(cu_seqlens, BT)
        NT = chunk.triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if core_attn_out is not None:
            if core_attn_out.numel() < v.numel():
                raise ValueError("core_attn_out is too small for HCU FLA chunk_o")
            out = core_attn_out[:v.numel()].view(*v.shape)
        else:
            out = chunk.torch.empty_like(v)
        launch_chunk_fwd_kernel_o(
            q=q, k=k, v=v, h=h, g=g, g_gamma=None, o=out,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale,
            T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, NT=NT, B=B,
            use_exp2=False, transpose_state_layout=True, kernel_cfg=None,
        )
        return out

    setattr(hcu_chunk_o, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_chunk_fwd_o", original)
    setattr(chunk, "chunk_fwd_o", hcu_chunk_o)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
