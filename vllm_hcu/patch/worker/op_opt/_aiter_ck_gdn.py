# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Audited access and input gating for AITER's fused CK GDN prefill."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from collections.abc import Callable, Mapping

import torch
from vllm.v1.worker.workspace import current_workspace_manager

from ._common import PatchCompatibilityError, require_exact_signature
from ._gdn_backend import AITER, requested_backend

_MODULE = "aiter"
_AUDITED_VERSION = "0.1.7"


@dataclass(frozen=True)
class AiterCkGdnOps:
    prefill: Callable
    workspace_bytes: Callable


_WORKSPACE_POSITIONAL = (
    "q", "v", "cu_seqlens", "use_g", "use_gk", "use_exp2",
    "transpose_state", "qk_l2norm", "gate_in_kernel", "has_dt_bias",
    "beta_sigmoid_in_kernel", "output_final_state", "cp_context",
    "cp_world_size",
)
_WORKSPACE_DEFAULTS = {
    "cu_seqlens": None, "use_g": True, "use_gk": False,
    "use_exp2": True, "transpose_state": False, "qk_l2norm": False,
    "gate_in_kernel": False, "has_dt_bias": False,
    "beta_sigmoid_in_kernel": False, "output_final_state": True,
    "cp_context": False, "cp_world_size": 2,
}
_PREFILL_POSITIONAL = (
    "q", "k", "v", "g", "beta", "workspace", "gk", "a_log",
    "dt_bias", "initial_state", "cu_seqlens", "scale", "use_g", "use_gk",
    "use_exp2", "transpose_state", "qk_l2norm", "gate_in_kernel",
    "has_dt_bias", "beta_sigmoid_in_kernel", "output_final_state",
    "cp_context", "cp_world_size", "cp_rank",
)
_PREFILL_DEFAULTS = {
    "workspace": None, "gk": None, "a_log": None, "dt_bias": None,
    "initial_state": None, "cu_seqlens": None, "scale": None,
    "use_g": True, "use_gk": False, "use_exp2": True,
    "transpose_state": False, "qk_l2norm": False,
    "gate_in_kernel": False, "has_dt_bias": False,
    "beta_sigmoid_in_kernel": False, "output_final_state": True,
    "cp_context": False, "cp_world_size": 2, "cp_rank": 1,
}


def resolve_aiter_ck_gdn() -> AiterCkGdnOps | None:
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name is None or not _MODULE.startswith(exc.name):
            raise
        return None
    prefill = getattr(module, "ck_gdn_prefill", None)
    workspace_bytes = getattr(module, "ck_gdn_prefill_workspace_bytes", None)
    if prefill is None or workspace_bytes is None:
        return None
    if not callable(prefill) or not callable(workspace_bytes):
        raise PatchCompatibilityError(
            "optional aiter CK GDN entry point exists but is not callable"
        )
    require_exact_signature(
        prefill, f"aiter.ck_gdn_prefill (audited aiter {_AUDITED_VERSION})",
        positional=_PREFILL_POSITIONAL,
        defaults=_PREFILL_DEFAULTS,
    )
    require_exact_signature(
        workspace_bytes,
        f"aiter.ck_gdn_prefill_workspace_bytes (audited aiter {_AUDITED_VERSION})",
        positional=_WORKSPACE_POSITIONAL,
        defaults=_WORKSPACE_DEFAULTS,
    )
    return AiterCkGdnOps(prefill=prefill, workspace_bytes=workspace_bytes)


def aiter_ck_unsupported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    capturing: bool,
) -> str | None:
    provider = requested_backend("prefill")
    if provider != AITER:
        return f"requested provider is {provider}"
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return "q, k, and v must be packed 3D tensors"
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return "key/value dimensions must both be 128"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "activations must be fp16 or bf16"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must share an activation dtype"
    hq, hv = q.shape[-2], v.shape[-2]
    if hq <= 0 or hv % hq or hv // hq not in (1, 2, 4):
        ratio = hv / hq if hq else 0
        return f"value/key head ratio {ratio:g} is unsupported"
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return "g and beta must be fp32"
    if g.shape != (q.shape[0], hv) or beta.shape != g.shape:
        return "g and beta shapes must match tokens and value heads"
    if not all(x.is_contiguous() for x in (q, k, v, g, beta)):
        return "CK inputs must be contiguous"
    sequences = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None:
        expected = (sequences, hv, 128, 128)
        if initial_state.dtype != torch.float32 or tuple(initial_state.shape) != expected:
            return "initial state must be contiguous fp32 [sequences, hv, 128, 128]"
        if not initial_state.is_contiguous():
            return "initial state must be contiguous fp32 [sequences, hv, 128, 128]"
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int64:
            return "cu_seqlens must be contiguous int64"
        if not cu_seqlens.is_contiguous():
            return "cu_seqlens must be contiguous int64"
        if capturing and cu_seqlens.numel() > 2:
            return "packed variable-length metadata is not graph-safe"
    return None


def aiter_ck_cu_seqlens(
    q: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor | None:
    del q
    if cu_seqlens is not None and cu_seqlens.numel() == 2:
        return None
    return cu_seqlens


def aiter_ck_workspace(
    ops: AiterCkGdnOps,
    q: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    flags: Mapping[str, object],
) -> torch.Tensor:
    required = int(ops.workspace_bytes(q, v, cu_seqlens, **flags))
    return current_workspace_manager().get_simultaneous(
        ((required,), torch.uint8)
    )[0]


__all__ = [
    "AiterCkGdnOps", "PatchCompatibilityError", "aiter_ck_cu_seqlens",
    "aiter_ck_unsupported", "aiter_ck_workspace", "resolve_aiter_ck_gdn",
]
