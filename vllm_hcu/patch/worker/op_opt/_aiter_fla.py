# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, ABI-checked access to optional self-developed AITER FLA kernels."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ._common import PatchCompatibilityError, require_exact_signature

_MODULE = "aiter.ops.fla"

_CONTRACTS: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
    "chunk_fwd_o_vllm_hip_blockdim64": (
        (
            "q", "k", "v", "h", "g", "g_gamma", "scale",
            "cu_seqlens", "chunk_size", "chunk_indices", "use_exp2",
            "transpose_state_layout", "kernel_cfg",
        ),
        {
            "g": None,
            "g_gamma": None,
            "scale": None,
            "cu_seqlens": None,
            "chunk_size": 64,
            "chunk_indices": None,
            "use_exp2": False,
            "transpose_state_layout": True,
            "kernel_cfg": None,
        },
    ),
}


def make_aiter_fla_resolver(name: str) -> Callable[[], Callable | None]:
    """Build a per-patch lazy resolver for an audited AITER FLA ABI."""
    positional, defaults = _CONTRACTS[name]
    resolved = False
    kernel: Callable | None = None

    def resolve() -> Callable | None:
        nonlocal kernel, resolved
        if resolved:
            return kernel
        try:
            module = importlib.import_module(_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name not in {"aiter", "aiter.ops", _MODULE}:
                raise
            resolved = True
            return None

        candidate = getattr(module, name, None)
        if candidate is None:
            resolved = True
            return None
        if not callable(candidate):
            raise PatchCompatibilityError(
                f"optional {_MODULE}.{name} exists but is not callable"
            )
        require_exact_signature(
            candidate,
            f"{_MODULE}.{name}",
            positional=positional,
            defaults=defaults,
        )
        kernel = candidate
        resolved = True
        return kernel

    return resolve


__all__ = ["make_aiter_fla_resolver"]
