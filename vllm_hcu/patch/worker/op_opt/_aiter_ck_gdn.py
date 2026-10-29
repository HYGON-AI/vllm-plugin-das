# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, ABI-checked access to the native aiter CK GDN entry points.

``hcu-aiter-ck`` runs the whole GDN prefill (kkt/solve/recompute/fwd_h/fwd_o)
as one ``aiter.ck_gdn_prefill`` call instead of the per-operator FLA pipeline,
and GDN decode as ``aiter.ck_gdn_decode``.  The audited contract is the
installed aiter 0.1.7 wheel (``aiter.ops.ck_gdn``), not the older 0.1.6 source
snapshot.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ._common import PatchCompatibilityError, require_exact_signature

_MODULE = "aiter.ops.ck_gdn"
_AUDITED_VERSION = "0.1.7"

_PREFILL_POSITIONAL = (
    "q", "k", "v", "g", "beta", "workspace", "gk", "a_log", "dt_bias",
    "initial_state", "cu_seqlens", "scale", "use_g", "use_gk", "use_exp2",
    "transpose_state", "qk_l2norm", "gate_in_kernel", "has_dt_bias",
    "beta_sigmoid_in_kernel", "output_final_state", "cp_context",
    "cp_world_size", "cp_rank",
)
_PREFILL_DEFAULTS: dict[str, object] = {
    "workspace": None,
    "gk": None,
    "a_log": None,
    "dt_bias": None,
    "initial_state": None,
    "cu_seqlens": None,
    "scale": None,
    "use_g": True,
    "use_gk": False,
    "use_exp2": True,
    "transpose_state": False,
    "qk_l2norm": False,
    "gate_in_kernel": False,
    "has_dt_bias": False,
    "beta_sigmoid_in_kernel": False,
    "output_final_state": True,
    "cp_context": False,
    "cp_world_size": 2,
    "cp_rank": 1,
}

_DECODE_POSITIONAL = (
    "q", "k", "v", "g", "beta", "gk", "gv", "a_log", "dt_bias",
    "initial_state", "cu_seqlens", "scale", "use_g", "use_gk", "use_gv",
    "beta_headwise", "use_qk_l2norm", "use_exp2", "transpose_state",
    "gate_in_kernel", "has_dt_bias", "store_final_state", "validate_varlen",
)
_DECODE_DEFAULTS: dict[str, object] = {
    "gk": None,
    "gv": None,
    "a_log": None,
    "dt_bias": None,
    "initial_state": None,
    "cu_seqlens": None,
    "scale": None,
    "use_g": True,
    "use_gk": False,
    "use_gv": False,
    "beta_headwise": True,
    "use_qk_l2norm": False,
    "use_exp2": False,
    "transpose_state": False,
    "gate_in_kernel": False,
    "has_dt_bias": False,
    "store_final_state": True,
    "validate_varlen": True,
}

_resolved: dict[str, Callable | None] = {}


def _resolve(
    attribute: str, positional: tuple[str, ...], defaults: dict[str, object]
) -> Callable | None:
    """Return ``aiter.ops.ck_gdn.<attribute>`` or ``None`` if absent.

    Raises :class:`PatchCompatibilityError` when the installed entry point
    does not match the audited signature.
    """

    if attribute in _resolved:
        return _resolved[attribute]
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name is None or not _MODULE.startswith(exc.name):
            raise
        _resolved[attribute] = None
        return None

    candidate = getattr(module, attribute, None)
    if candidate is None:
        _resolved[attribute] = None
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
    _resolved[attribute] = candidate
    return candidate


def resolve_ck_gdn_prefill() -> Callable | None:
    return _resolve("ck_gdn_prefill", _PREFILL_POSITIONAL, _PREFILL_DEFAULTS)


def resolve_ck_gdn_decode() -> Callable | None:
    return _resolve("ck_gdn_decode", _DECODE_POSITIONAL, _DECODE_DEFAULTS)


__all__ = ["resolve_ck_gdn_decode", "resolve_ck_gdn_prefill"]
