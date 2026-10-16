# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Resolve which provider runs each HCU GDN kernel.

Prefill is selected by ``--gdn-prefill-backend`` and decode by
``VLLM_HCU_GDN_DECODE_BACKEND``; both accept ``triton`` (the vLLM FLA kernel,
no HCU replacement), ``hcu-boltops`` and ``hcu-aiter``.  The HCU default for
both stages is ``hcu-aiter``.  ``VLLM_HCU_USE_CUSTOM_OPS=0`` forces
``hcu-boltops``.

A provider that cannot run an operator (package not installed or no such
kernel) falls back with a warning: ``hcu-aiter -> hcu-boltops -> triton`` and
``hcu-boltops -> triton``.  An installed kernel whose signature differs from
the audited contract raises ``PatchCompatibilityError`` instead: that is a
version mismatch to fix, not a missing optional package.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Literal

from vllm.logger import init_logger

from vllm_hcu.patch.config import HCU_GDN_PREFILL_BACKENDS, get_hcu_config

# Under the ``vllm`` logger tree so INFO selection lines reach vLLM's handler.
logger = init_logger(f"vllm.{__name__}")

Stage = Literal["prefill", "decode"]
Resolver = Callable[[], Callable | None]

TRITON = "triton"
BOLTOPS = "hcu-boltops"
AITER = "hcu-aiter"
BACKENDS = (TRITON, BOLTOPS, AITER)
DEFAULT_BACKEND = AITER
_FALLBACK_CHAINS = {
    TRITON: (TRITON,),
    BOLTOPS: (BOLTOPS, TRITON),
    AITER: (AITER, BOLTOPS, TRITON),
}
_LEGACY_ENV = "VLLM_HCU_USE_CUSTOM_AITER_FLA"
_UPSTREAM_AUTO = "auto"

# The prefill selection is captured from VllmConfig while the model is
# constructed (see patch_gdn_linear_attention); FLA kernels later run outside
# any set_current_vllm_config() context.
_prefill_requested: str | None = None


def requested_prefill_backend(vllm_config: object) -> str:
    """Map the public ``--gdn-prefill-backend`` value to an HCU backend."""

    hcu_backend = get_hcu_config(vllm_config).hcu_gdn_prefill_backend
    if hcu_backend is not None:
        return hcu_backend
    additional = getattr(vllm_config, "additional_config", None)
    upstream = _UPSTREAM_AUTO
    if isinstance(additional, Mapping):
        upstream = str(
            additional.get("gdn_prefill_backend", _UPSTREAM_AUTO)
        ).strip().lower()
    if upstream == TRITON:
        return TRITON
    if upstream != _UPSTREAM_AUTO:
        logger.warning_once(
            "GDN prefill backend %r is not available on HCU; using %s.",
            upstream,
            DEFAULT_BACKEND,
        )
    return DEFAULT_BACKEND


def record_prefill_backend(vllm_config: object) -> str:
    """Remember the prefill selection of the model being constructed."""

    global _prefill_requested
    _prefill_requested = requested_prefill_backend(vllm_config)
    return _prefill_requested


def _prefill_backend() -> str:
    if _prefill_requested is not None:
        return _prefill_requested
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return record_prefill_backend(vllm_config)
    return DEFAULT_BACKEND


def _decode_backend() -> str:
    from vllm_hcu.platforms import envs as henvs

    backend = henvs.VLLM_HCU_GDN_DECODE_BACKEND
    if backend in BACKENDS:
        return backend
    logger.warning_once(
        "Invalid VLLM_HCU_GDN_DECODE_BACKEND=%r (expected one of: %s); "
        "using %s.",
        backend,
        ", ".join(BACKENDS),
        DEFAULT_BACKEND,
    )
    return DEFAULT_BACKEND


def requested_backend(stage: Stage) -> str:
    """Return the requested backend for ``stage`` under the master gate."""

    from vllm_hcu.platforms import envs as henvs

    if _LEGACY_ENV in os.environ:
        logger.warning_once(
            "%s is deprecated and ignored; use --gdn-prefill-backend and "
            "VLLM_HCU_GDN_DECODE_BACKEND instead.",
            _LEGACY_ENV,
        )
    if not henvs.custom_ops_enabled():
        return BOLTOPS
    if stage == "prefill":
        return _prefill_backend()
    return _decode_backend()


def make_provider_selector(
    stage: Stage,
    op: str,
    resolvers: Mapping[str, Resolver],
) -> Callable[[], tuple[str, Callable | None]]:
    """Return a selector yielding ``(provider, kernel)`` for ``op``.

    ``resolvers`` maps ``hcu-boltops``/``hcu-aiter`` to lazy kernel
    resolvers; a provider without an entry has no kernel for ``op``.
    ``triton`` carries no kernel: the caller runs the original vLLM
    function.  The choice is cached per requested backend.
    """

    selected: dict[str, tuple[str, Callable | None]] = {}

    def select() -> tuple[str, Callable | None]:
        requested = requested_backend(stage)
        cached = selected.get(requested)
        if cached is not None:
            return cached
        provider, kernel = TRITON, None
        for candidate in _FALLBACK_CHAINS[requested]:
            if candidate == TRITON:
                break
            resolve = resolvers.get(candidate)
            if resolve is None:
                reason = "it has no kernel for this operator"
            else:
                kernel = resolve()
                reason = "the package or kernel is not installed"
            if kernel is not None:
                provider = candidate
                break
            logger.warning_once(
                "GDN %s %s: %s cannot be used (requested %s) because %s; "
                "falling back.",
                stage,
                op,
                candidate,
                requested,
                reason,
            )
        logger.info_once(
            "GDN %s %s uses %s (requested %s).", stage, op, provider, requested
        )
        selected[requested] = (provider, kernel)
        return provider, kernel

    return select


def aiter_input_fallback(
    stage: Stage,
    op: str,
    reason: str,
    resolve_boltops: Resolver,
) -> tuple[str, Callable | None]:
    """Fall back for one call whose inputs the aiter HIP kernel rejects.

    Follows the ``hcu-aiter`` chain: ``hcu-boltops``, then ``triton``.
    """

    kernel = resolve_boltops()
    provider = BOLTOPS if kernel is not None else TRITON
    logger.warning_once(
        "GDN %s %s: hcu-aiter does not support %s; using %s for these inputs.",
        stage,
        op,
        reason,
        provider,
    )
    return provider, kernel


__all__ = [
    "AITER",
    "BACKENDS",
    "BOLTOPS",
    "DEFAULT_BACKEND",
    "TRITON",
    "aiter_input_fallback",
    "make_provider_selector",
    "record_prefill_backend",
    "requested_backend",
    "requested_prefill_backend",
]

assert set(HCU_GDN_PREFILL_BACKENDS) == {BOLTOPS, AITER}
