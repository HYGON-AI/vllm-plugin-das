# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, ABI-checked access to optional aiter HIP GDN kernels.

``hcu-aiter`` uses the HIP kernels in ``aiter.ops.fla``; they replace the
vLLM FLA functions one-to-one and accept the framework's precomputed
``chunk_indices``/``chunk_offsets``.  aiter has no ``recompute_w_u_fwd``.

The HIP decode kernels treat only negative state indices as invalid, unlike
vLLM's NULL_BLOCK_ID=0 convention; callers must remap padding indices.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ._common import PatchCompatibilityError, require_exact_signature

_MODULE = "aiter.ops.fla"
_AUDITED_VERSION = "0.1.6"

# name -> (attribute, positional parameters, defaults)
_CONTRACTS: dict[str, tuple[str, tuple[str, ...], dict[str, object]]] = {
    "chunk_gated_delta_rule_fwd_h": (
        "chunk_gated_delta_rule_fwd_vllm_hip_blockdim64",
        (
            "k", "w", "u", "g", "gk", "initial_state",
            "initial_state_indices", "output_final_state", "chunk_size",
            "save_new_value", "cu_seqlens", "chunk_indices",
            "chunk_offsets", "use_exp2", "transpose_state_layout",
            "kernel_cfg",
        ),
        {
            "g": None,
            "gk": None,
            "initial_state": None,
            "initial_state_indices": None,
            "output_final_state": True,
            "chunk_size": 64,
            "save_new_value": True,
            "cu_seqlens": None,
            "chunk_indices": None,
            "chunk_offsets": None,
            "use_exp2": False,
            "transpose_state_layout": True,
            "kernel_cfg": None,
        },
    ),
    "chunk_fwd_o": (
        "chunk_fwd_o_vllm_hip_blockdim64",
        (
            "q", "k", "v", "h", "g", "g_gamma", "scale", "cu_seqlens",
            "chunk_size", "chunk_indices", "use_exp2",
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
    "fused_sigmoid_gating_delta_rule_update": (
        "vllm_fused_sigmoid_gating_delta_rule_update",
        (
            "A_log", "a", "b", "dt_bias", "q", "k", "v", "beta",
            "threshold", "scale", "initial_state", "inplace_final_state",
            "cu_seqlens", "ssm_state_indices", "num_accepted_tokens",
            "use_qk_l2norm_in_kernel", "is_kda",
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
        },
    ),
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "aiter_fused_recurrent_gated_delta_rule_packed_decode",
        (
            "mixed_qkv", "a", "b", "A_log", "dt_bias", "scale",
            "initial_state", "out", "ssm_state_indices",
            "use_qk_l2norm_in_kernel", "kernel_cfg",
        ),
        {"use_qk_l2norm_in_kernel": False, "kernel_cfg": None},
    ),
}


def make_aiter_gdn_resolver(name: str) -> Callable[[], Callable | None]:
    """Build a lazy resolver for the audited aiter HIP kernel behind ``name``.

    Returns ``None`` when aiter is not installed or lacks the kernel and
    raises :class:`PatchCompatibilityError` when the installed entry point
    does not match the audited contract.
    """

    attribute, positional, defaults = _CONTRACTS[name]
    resolved = False
    kernel: Callable | None = None

    def resolve() -> Callable | None:
        nonlocal kernel, resolved
        if resolved:
            return kernel
        try:
            module = importlib.import_module(_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name is None or not _MODULE.startswith(exc.name):
                raise
            resolved = True
            return None

        candidate = getattr(module, attribute, None)
        if candidate is None:
            resolved = True
            return None
        if not callable(candidate):
            raise PatchCompatibilityError(
                f"optional {_MODULE}.{attribute} exists but is not callable"
            )
        require_exact_signature(
            candidate,
            f"{_MODULE}.{attribute} (audited aiter {_AUDITED_VERSION})",
            positional=positional,
            defaults=defaults,
        )
        kernel = candidate
        resolved = True
        return kernel

    return resolve


__all__ = ["make_aiter_gdn_resolver"]
