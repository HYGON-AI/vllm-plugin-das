# SPDX-License-Identifier: Apache-2.0
"""Select the HCU AITER gated-delta state kernel without source mutation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.model_executor.layers.fla.ops.chunk_delta_h"
PATCH_ID = "worker.op_opt.fla.chunk_delta_h.aiter"
TARGETS = (f"{TARGET_MODULE}.chunk_gated_delta_rule_fwd_h",)
_MARKER = "_vllm_hcu_chunk_delta_h_applied"
_WRAPPER = "_vllm_hcu_chunk_delta_h_wrapper"


def _enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA and henvs.VLLM_HCU_USE_CUSTOM_OPS)


def apply_to_module(module: ModuleType) -> bool:
    chunk = load_exact_module(TARGET_MODULE, module)
    if already_applied(chunk, _MARKER, ((chunk, "chunk_gated_delta_rule_fwd_h", TARGETS[0], _WRAPPER),)):
        return False
    original = require_callable(chunk, "chunk_gated_delta_rule_fwd_h", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "k", "w", "u", "g", "gk", "initial_state", "output_final_state",
            "chunk_size", "save_new_value", "cu_seqlens", "chunk_indices", "chunk_offsets",
        ),
        defaults={
            "g": None, "gk": None, "initial_state": None,
            "output_final_state": False, "chunk_size": chunk.FLA_CHUNK_SIZE,
            "save_new_value": True, "cu_seqlens": None, "chunk_indices": None,
            "chunk_offsets": None,
        },
    )

    @functools.wraps(original)
    def hcu_chunk_delta_h(
        k, w, u, g=None, gk=None, initial_state=None, output_final_state=False,
        chunk_size=chunk.FLA_CHUNK_SIZE, save_new_value=True, cu_seqlens=None,
        chunk_indices=None, chunk_offsets=None,
    ):
        if not _enabled():
            return original(k, w, u, g, gk, initial_state, output_final_state,
                            chunk_size, save_new_value, cu_seqlens, chunk_indices,
                            chunk_offsets)
        try:
            from aiter.ops.triton.fla.vllm.chunk_delta_h import (
                launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            )
        except ImportError as exc:
            raise RuntimeError("HCU AITER FLA chunk_delta_h is enabled but unavailable") from exc

        B, T, Hg, K, V = *k.shape, u.shape[-1]
        H, BT = u.shape[-2], chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = chunk.prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is None:
            N, NT, chunk_offsets = B, chunk.triton.cdiv(T, BT), None
        else:
            N, NT = len(cu_seqlens) - 1, len(chunk_indices)
            if chunk_offsets is None:
                chunk_offsets = chunk.prepare_chunk_offsets(cu_seqlens, BT)
        if K > 256:
            raise ValueError("HCU AITER FLA does not support head dimension > 256")
        h = k.new_empty(B, NT, H, V, K)
        final_state = (
            k.new_empty(N, H, V, K, dtype=chunk.torch.float32)
            if output_final_state else None
        )
        v_new = chunk.torch.empty_like(u) if save_new_value else None
        launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
            k=k, u=u, w=w, v_new=v_new, g=g, gk=gk, h=h,
            initial_state=initial_state, initial_state_indices=None,
            final_state=final_state, cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets, N=N, T=T, H=H, Hg=Hg, K=K, V=V,
            BT=BT, use_exp2=False, transpose_state_layout=True, kernel_cfg=None,
        )
        return h, v_new, final_state

    setattr(hcu_chunk_delta_h, _WRAPPER, True)
    setattr(chunk, "_vllm_hcu_original_chunk_gated_delta_rule_fwd_h", original)
    setattr(chunk, "chunk_gated_delta_rule_fwd_h", hcu_chunk_delta_h)
    setattr(chunk, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
