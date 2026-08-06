# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light KV-cache layout helpers shared by runtime and tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def split_kv_cache(
    kv_cache: object,
    *,
    kv_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return key and value tensors from split or explicitly stacked storage."""
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(f"expected two split KV cache tensors, got {len(kv_cache)}")
        return kv_cache[0], kv_cache[1]
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")
    if not -kv_cache.ndim <= kv_axis < kv_cache.ndim:
        raise ValueError(
            f"KV cache axis {kv_axis} is out of range for shape "
            f"{tuple(kv_cache.shape)}"
        )
    if kv_cache.shape[kv_axis] == 2:
        return kv_cache.unbind(kv_axis)
    raise ValueError(
        "expected stacked KV cache dimension of size 2 at "
        f"axis {kv_axis}, "
        f"got shape {tuple(kv_cache.shape)}"
    )


def has_mixed_kv_cache_block_dims(
    groups: Iterable[Any],
    kernel_block_sizes: list[int],
    cache_dtype_str: str,
) -> bool:
    """Return whether attention groups disagree on their physical block axis."""

    block_dims: set[int] = set()
    for group in groups:
        group_id = group.kv_cache_group_id
        if group_id == len(kernel_block_sizes):
            continue
        spec = group.kv_cache_spec
        block_dims.add(
            group.backend.get_kv_cache_block_dim(
                kernel_block_sizes[group_id],
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype_str,
            )
        )
    return len(block_dims) > 1


__all__ = ["has_mixed_kv_cache_block_dims", "split_kv_cache"]
