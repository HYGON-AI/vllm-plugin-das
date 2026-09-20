# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""QSA-specific kernel dispatch.

QSA is owned by the Qwen4Exp model and is intentionally not registered as a
global ``AttentionBackendEnum`` member.  This module only chooses the QSA
kernel pair used by that model:

* ``TRITON_QSA`` keeps the official QSA implementation;
* ``FLASH_QSA`` uses flash_attn's QSA entry points in HCU CUTLASS mode.

The generic ``FLASH_ATTN`` backend remains responsible for ordinary attention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal


QSA_BACKEND_TRITON: Literal["TRITON_QSA"] = "TRITON_QSA"
QSA_BACKEND_FLASH: Literal["FLASH_QSA"] = "FLASH_QSA"
QSAKernelName = Literal["TRITON_QSA", "FLASH_QSA"]
_QSA_MODES = frozenset(("classic", "cutlass", "varlen"))


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
            "FLASH_ATTN_CUTLASS QSA requires flash_attn's "
            "mqa_paged_score_func and sparse_gqa_paged_attn_func"
        ) from exc

    if not callable(mqa_paged_score_func) or not callable(
        sparse_gqa_paged_attn_func
    ):
        raise RuntimeError(
            "flash_attn QSA entry points are not callable"
        )
    return mqa_paged_score_func, sparse_gqa_paged_attn_func


def get_qsa_flash_attn_mode() -> str:
    """Return the mode captured by the current forward context when present.

    ``set_forward_context`` receives the worker's deserialised ``VllmConfig``
    while ``get_current_vllm_config_or_none`` is not guaranteed to be set in a
    Model Runner V2 forward.  HCU stores the resolved mode in
    ``additional_kwargs`` for this reason.  The platform resolver remains a
    fallback for unit tests and non-forward calls.
    """

    try:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
    except (AssertionError, ImportError):
        context = None

    if context is not None:
        additional_kwargs = getattr(context, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            mode = additional_kwargs.get("hcu_flash_attn_mode")
            if isinstance(mode, str) and mode in _QSA_MODES:
                return mode

    from vllm_hcu.platforms.hcu import get_hcu_flash_attn_mode

    return get_hcu_flash_attn_mode()


def get_qsa_kernel_backend(
    mode: str,
    *,
    triton_mqa_paged: Callable[..., Any],
    triton_sparse_gqa_paged_attn: Callable[..., Any],
) -> QSAKernelBackend:
    """Build the QSA kernel adapter for one normalized HCU flash mode.

    Only CUTLASS uses flash_attn's QSA kernels.  Classic and varlen retain the
    official Triton QSA path because the flash_attn QSA entry points are not
    generic replacements for those implementations.
    """

    if mode != "cutlass":
        return QSAKernelBackend(
            name=QSA_BACKEND_TRITON,
            mqa_paged_score=triton_mqa_paged,
            sparse_gqa_paged_attn=triton_sparse_gqa_paged_attn,
        )

    flash_mqa, flash_sparse = _load_flash_qsa_kernels()
    return QSAKernelBackend(
        name=QSA_BACKEND_FLASH,
        mqa_paged_score=flash_mqa,
        sparse_gqa_paged_attn=flash_sparse,
    )


__all__ = [
    "QSA_BACKEND_FLASH",
    "QSA_BACKEND_TRITON",
    "QSAKernelBackend",
    "get_qsa_flash_attn_mode",
    "get_qsa_kernel_backend",
]
