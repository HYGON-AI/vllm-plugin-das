# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional LightOp fused top-k and token transform for pooled sparse MLA."""

from __future__ import annotations

from functools import cache
from types import ModuleType
from typing import Callable

import torch

from vllm_hcu.platforms import envs as henvs


@cache
def _get_lightop_kpool_transform() -> Callable | None:
    try:
        from lightop.fuse_topk_transform import (
            fast_kpool_topk_transform_fused,
        )
    except (ImportError, AttributeError):
        return None
    return fast_kpool_topk_transform_fused


def lightop_kpool_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    *,
    page_table: torch.Tensor | None = None,
    topk_indices_offset: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK
    ):
        return None
    if pool_size not in (4, 16) or topk != 2048 or seq_lens is None:
        return None
    fused = _get_lightop_kpool_transform()
    if fused is None:
        return None
    return fused(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        seq_lens=seq_lens,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )


def install_lightop_kpool_topk_transform(kpool_module: ModuleType) -> bool:
    register = getattr(
        kpool_module,
        "register_kpool_topk_transform_fused",
        None,
    )
    if not callable(register):
        return False
    register(lightop_kpool_topk_transform)
    return True


__all__ = [
    "install_lightop_kpool_topk_transform",
    "lightop_kpool_topk_transform",
]
