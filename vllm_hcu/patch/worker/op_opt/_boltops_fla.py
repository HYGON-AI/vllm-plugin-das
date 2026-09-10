# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, ABI-checked access to optional BoltOPs GDN kernels."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ._common import PatchCompatibilityError, require_exact_signature

_MODULE = "boltops.fla.gdn"
_AUDITED_VERSION = "0.1.0"

_CONTRACTS: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
    "chunk_gated_delta_rule_fwd_h": (
        (
            "k", "w", "u", "g", "gk", "initial_state",
            "initial_state_indices", "output_final_state",
            "inplace_final_state", "chunk_size", "save_new_value",
            "cu_seqlens", "chunk_indices", "use_exp2",
            "transpose_state_layout", "kernel_cfg",
        ),
        {
            "g": None,
            "gk": None,
            "initial_state": None,
            "initial_state_indices": None,
            "output_final_state": True,
            "inplace_final_state": False,
            "chunk_size": 64,
            "save_new_value": True,
            "cu_seqlens": None,
            "chunk_indices": None,
            "use_exp2": False,
            "transpose_state_layout": True,
            "kernel_cfg": None,
        },
    ),
    "chunk_fwd_o": (
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
            "transpose_state_layout": False,
            "kernel_cfg": None,
        },
    ),
    "recompute_w_u_fwd": (
        (
            "k", "v", "beta", "g_cumsum", "A", "cu_seqlens",
            "chunk_indices",
        ),
        {"cu_seqlens": None, "chunk_indices": None},
    ),
    "fused_sigmoid_gating_delta_rule_update": (
        (
            "A_log", "a", "b", "dt_bias", "q", "k", "v", "beta",
            "threshold", "scale", "initial_state", "inplace_final_state",
            "cu_seqlens", "ssm_state_indices", "num_accepted_tokens",
            "use_qk_l2norm_in_kernel", "is_kda", "kernel_cfg",
        ),
        {
            "beta": 1.0,
            "threshold": 20.0,
            "scale": None,
            "initial_state": None,
            "inplace_final_state": True,
            "cu_seqlens": None,
            "ssm_state_indices": None,
            "num_accepted_tokens": None,
            "use_qk_l2norm_in_kernel": False,
            "is_kda": False,
            "kernel_cfg": None,
        },
    ),
    "fused_recurrent_gated_delta_rule_packed_decode": (
        (
            "mixed_qkv", "a", "b", "A_log", "dt_bias", "scale",
            "initial_state", "out", "ssm_state_indices",
            "use_qk_l2norm_in_kernel", "kernel_cfg",
        ),
        {"use_qk_l2norm_in_kernel": False, "kernel_cfg": None},
    ),
}


def make_boltops_gdn_resolver(name: str) -> Callable[[], Callable | None]:
    """Build a per-patch lazy resolver for an audited public BoltOPs ABI."""
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
            if exc.name not in {"boltops", "boltops.fla", _MODULE}:
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
            f"{_MODULE}.{name} (audited BoltOPs {_AUDITED_VERSION})",
            positional=positional,
            defaults=defaults,
        )
        kernel = candidate
        resolved = True
        return kernel

    return resolve


__all__ = ["make_boltops_gdn_resolver"]
