# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""QSA-specific kernel dispatch.

QSA is owned by the Qwen4Exp model and is intentionally not registered as a
global ``AttentionBackendEnum`` member.  This module only chooses the QSA
kernel pair used by that model:

* ``TRITON_QSA`` keeps the official QSA implementation;
* ``FLASH_QSA`` uses flash_attn's QSA entry points when both
  ``VLLM_HCU_USE_CUSTOM_OPS`` and ``VLLM_HCU_USE_QSA_CUTLASS`` are enabled.

The generic ``FLASH_ATTN`` backend remains responsible for ordinary attention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import vllm_hcu.platforms.envs as henvs


QSA_BACKEND_TRITON: Literal["TRITON_QSA"] = "TRITON_QSA"
QSA_BACKEND_FLASH: Literal["FLASH_QSA"] = "FLASH_QSA"
QSAKernelName = Literal["TRITON_QSA", "FLASH_QSA"]


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


def is_qsa_cutlass_enabled() -> bool:
    """Return whether FA QSA is enabled by both HCU switches."""

    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_QSA_CUTLASS
    )


def get_qsa_kernel_backend(
    *,
    triton_mqa_paged: Callable[..., Any],
    triton_sparse_gqa_paged_attn: Callable[..., Any],
) -> QSAKernelBackend:
    """Build the QSA adapter from the QSA-specific environment switch.

    QSA selection is independent of the generic FLASH_ATTN backend mode. The
    global custom-op switch remains the master gate.
    """

    if not is_qsa_cutlass_enabled():
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
    "get_qsa_kernel_backend",
    "is_qsa_cutlass_enabled",
]
