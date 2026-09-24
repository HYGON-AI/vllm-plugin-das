# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""QSA-specific kernel dispatch.

QSA is owned by the Qwen4Exp model and is intentionally not registered as a
global ``AttentionBackendEnum`` member.  This module only chooses the QSA
kernel pair used by that model:

* ``triton`` keeps the official QSA implementation;
* ``cutlass`` uses flash_attn's QSA entry points;
* ``boltops`` uses BoltOPs' QSA entry points.

The generic ``FLASH_ATTN`` backend remains responsible for ordinary attention.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import vllm_hcu.platforms.envs as henvs

logger = logging.getLogger(__name__)


QSA_BACKEND_TRITON: Literal["triton"] = "triton"
QSA_BACKEND_CUTLASS: Literal["cutlass"] = "cutlass"
QSA_BACKEND_BOLTOPS: Literal["boltops"] = "boltops"
QSAKernelName = Literal["triton", "cutlass", "boltops"]
_QSA_BACKENDS = frozenset(
    (QSA_BACKEND_TRITON, QSA_BACKEND_CUTLASS, QSA_BACKEND_BOLTOPS)
)
_LEGACY_QSA_CUTLASS_ENV = "VLLM_HCU_USE_QSA_CUTLASS"
_legacy_qsa_cutlass_warning_emitted = False


@dataclass(frozen=True, slots=True)
class QSAKernelBackend:
    """The two compute entry points required by QSA."""

    name: QSAKernelName
    mqa_paged_score: Callable[..., Any]
    sparse_gqa_paged_attn: Callable[..., Any]


@lru_cache(maxsize=1)
def _load_flash_qsa_kernels() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load the optional flash_attn QSA entry points once per worker."""

    try:
        from flash_attn import (
            mqa_paged_score_func,
            sparse_gqa_paged_attn_func,
        )
    except Exception as exc:
        raise RuntimeError(
            "QSA FlashAttention requires flash_attn's "
            "mqa_paged_score_func and sparse_gqa_paged_attn_func"
        ) from exc

    if not callable(mqa_paged_score_func) or not callable(
        sparse_gqa_paged_attn_func
    ):
        raise RuntimeError(
            "flash_attn QSA entry points are not callable"
        )
    return mqa_paged_score_func, sparse_gqa_paged_attn_func


@lru_cache(maxsize=1)
def _load_boltops_qsa_kernels() -> tuple[
    Callable[..., Any], Callable[..., Any]
]:
    """Load the optional BoltOPs QSA entry points once per worker."""

    try:
        from boltops.qsa import (
            qsa_mqa_paged,
            qsa_sparse_paged_attention,
        )
    except Exception as exc:
        raise RuntimeError(
            "QSA BoltOPs requires boltops.qsa.qsa_mqa_paged and "
            "boltops.qsa.qsa_sparse_paged_attention"
        ) from exc

    if not callable(qsa_mqa_paged) or not callable(qsa_sparse_paged_attention):
        raise RuntimeError("boltops.qsa QSA entry points are not callable")
    return qsa_mqa_paged, qsa_sparse_paged_attention


def _warn_legacy_qsa_cutlass_env() -> None:
    global _legacy_qsa_cutlass_warning_emitted
    if (
        _LEGACY_QSA_CUTLASS_ENV in os.environ
        and not _legacy_qsa_cutlass_warning_emitted
    ):
        logger.warning(
            "%s is deprecated and ignored; use VLLM_HCU_QSA_BACKEND="
            "cutlass instead",
            _LEGACY_QSA_CUTLASS_ENV,
        )
        _legacy_qsa_cutlass_warning_emitted = True


def _resolve_qsa_backend() -> QSAKernelName:
    """Resolve the configured QSA backend, including the master gate."""

    _warn_legacy_qsa_cutlass_env()
    if not henvs.VLLM_HCU_USE_CUSTOM_OPS:
        return QSA_BACKEND_TRITON

    backend = henvs.VLLM_HCU_QSA_BACKEND
    if backend not in _QSA_BACKENDS:
        supported = ", ".join(sorted(_QSA_BACKENDS))
        raise ValueError(
            f"Invalid VLLM_HCU_QSA_BACKEND={backend!r}; "
            f"expected one of: {supported}"
        )
    return backend


def _triton_qsa_backend(
    triton_mqa_paged: Callable[..., Any],
    triton_sparse_gqa_paged_attn: Callable[..., Any],
) -> QSAKernelBackend:
    return QSAKernelBackend(
        name=QSA_BACKEND_TRITON,
        mqa_paged_score=triton_mqa_paged,
        sparse_gqa_paged_attn=triton_sparse_gqa_paged_attn,
    )


def _selected_qsa_loader(backend: QSAKernelName) -> Callable[[], tuple]:
    if backend == QSA_BACKEND_CUTLASS:
        return _load_flash_qsa_kernels
    if backend == QSA_BACKEND_BOLTOPS:
        return _load_boltops_qsa_kernels
    raise AssertionError(f"unexpected non-Triton QSA backend: {backend}")


# Caches the usable kernel pair, so that an unavailable optional backend
# neither re-imports, re-warns, nor raises per decode step. The outcome of a
# failed load is recorded as ``None`` rather than the exception itself:
# re-raising a cached exception would append this frame to its traceback on
# every forward pass, and the caller would only fall back to Triton anyway.
# Keyed by the loader object as well as the backend name so that replacing a
# loader (e.g. under test) cannot reuse a stale outcome.
_selected_qsa_cache: dict[
    tuple[QSAKernelName, Callable[[], tuple]],
    tuple[tuple[Callable[..., Any], Callable[..., Any]] | None],
] = {}


@lru_cache(maxsize=1)
def _load_flash_qsa_fp8_kernel() -> Callable[..., Any]:
    """Load and validate the optional FlashAttention FP8 QSA reader."""

    try:
        from flash_attn import sparse_gqa_paged_attn_fp8_func
        from flash_attn.flash_attn_interface import flash_attn_cuda
    except Exception as exc:
        raise RuntimeError(
            "QSA FP8 FlashAttention requires "
            "sparse_gqa_paged_attn_fp8_func"
        ) from exc

    native = getattr(flash_attn_cuda, "sparse_gqa_paged_attention_fp8", None)
    if not callable(sparse_gqa_paged_attn_fp8_func) or not callable(native):
        raise RuntimeError(
            "QSA FP8 FlashAttention Python or extension symbol is missing"
        )
    return sparse_gqa_paged_attn_fp8_func


_selected_qsa_fp8_cache: dict[
    Callable[[], Callable[..., Any]], tuple[Callable[..., Any] | None]
] = {}


def get_qsa_fp8_reader(
    *,
    triton_fp8: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """Resolve the FP8 main-cache reader before CUDA Graph capture.

    The QSA indexer and its paged MQA score kernel remain BF16. Only CUTLASS
    currently supplies an HCU-validated FP8 sparse-GQA reader; other backend
    selections retain an upstream Triton FP8 reader when one is available.
    """

    backend = _resolve_qsa_backend()
    if backend != QSA_BACKEND_CUTLASS:
        if callable(triton_fp8):
            return triton_fp8
        raise RuntimeError(
            f"QSA backend {backend!r} has no FP8 sparse-GQA reader"
        )

    loader = _load_flash_qsa_fp8_kernel
    cached = _selected_qsa_fp8_cache.get(loader)
    if cached is None:
        try:
            reader: Callable[..., Any] | None = loader()
        except Exception as exc:
            reader = None
            logger.warning(
                "QSA CUTLASS FP8 reader failed to load; falling back to "
                "Triton: %s",
                exc,
            )
        cached = (reader,)
        _selected_qsa_fp8_cache[loader] = cached

    reader = cached[0]
    if reader is not None:
        return reader
    if callable(triton_fp8):
        return triton_fp8
    raise RuntimeError(
        "QSA FP8 requires FlashAttention's paged FP8 symbol or an upstream "
        "Triton FP8 reader"
    )


def _load_selected_qsa_backend(
    backend: QSAKernelName,
) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Load an optional backend, warning once and returning None on failure.

    The wrappers call this on every forward pass, so the import attempt, its
    warning, and the fallback decision are all one-shot.
    """
    loader = _selected_qsa_loader(backend)
    key = (backend, loader)
    cached = _selected_qsa_cache.get(key)
    if cached is None:
        try:
            kernels: tuple[Callable[..., Any], Callable[..., Any]] | None = (
                loader()
            )
        except Exception as exc:
            kernels = None
            logger.warning(
                "QSA backend %r failed to load; falling back to Triton: %s",
                backend,
                exc,
            )
        cached = (kernels,)
        _selected_qsa_cache[key] = cached
    return cached[0]


def get_qsa_kernel_backend(
    *,
    triton_mqa_paged: Callable[..., Any],
    triton_sparse_gqa_paged_attn: Callable[..., Any],
) -> QSAKernelBackend:
    """Build the QSA adapter from the QSA-specific environment switch.

    QSA selection is independent of the generic FLASH_ATTN backend mode. The
    global custom-op switch remains the master gate. Optional backend loading
    failures warn and fall back to the official Triton implementation.
    """

    backend = _resolve_qsa_backend()
    triton_backend = _triton_qsa_backend(
        triton_mqa_paged,
        triton_sparse_gqa_paged_attn,
    )
    if backend == QSA_BACKEND_TRITON:
        return triton_backend

    selected = _load_selected_qsa_backend(backend)
    if selected is None:
        return triton_backend
    selected_mqa, selected_sparse = selected
    return QSAKernelBackend(
        name=backend,
        mqa_paged_score=selected_mqa,
        sparse_gqa_paged_attn=selected_sparse,
    )


__all__ = [
    "QSA_BACKEND_BOLTOPS",
    "QSA_BACKEND_CUTLASS",
    "QSA_BACKEND_TRITON",
    "QSAKernelBackend",
    "get_qsa_fp8_reader",
    "get_qsa_kernel_backend",
]
