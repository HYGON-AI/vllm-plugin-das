# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Utilities for sparse-MLA decode Top-K selection and result staging.

The sparse indexer stores decode and prefill results in one logical output
buffer.  A variable-length decode batch is temporarily expanded to a
rectangular ``[batch, max_decode_len]`` layout, so its Top-K kernel can produce
more rows than there are actual decode tokens.  The padded rows must never be
written into the shared output buffer: the rows immediately following the
decode prefix belong to prefill. The module also owns optional HCU LightOp DCP
selection helpers so whole-module replacements can keep the upstream static
definition surface.
"""

from __future__ import annotations

import functools

import torch

from vllm_hcu.platforms import envs as henvs


_LIGHTOP_DCP_TOPK_METADATA: dict[
    tuple[str, int | None, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@functools.lru_cache(maxsize=1)
def get_lightop_fast_topk_transform():
    """Resolve the optional categorized LightOp fused TopK API."""
    try:
        from lightop.attention import fast_topk_transform_fused
    except (AttributeError, ImportError, OSError):
        return None
    return (
        fast_topk_transform_fused
        if callable(fast_topk_transform_fused)
        else None
    )


def use_lightop_dcp_topk_transform() -> bool:
    """Return whether the LightOp DCP candidate selector is enabled."""
    return (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK
        and henvs.VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM
    )


def get_lightop_dcp_topk_metadata(
    device: torch.device,
    rows: int,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable, capacity-bucketed metadata for DCP TopK."""
    device = torch.device(device)
    capacity = 1 << (max(rows, 1) - 1).bit_length()
    key = (device.type, device.index, candidate_count, capacity)
    metadata = _LIGHTOP_DCP_TOPK_METADATA.get(key)
    if metadata is None:
        lengths = torch.full(
            (capacity,), candidate_count, dtype=torch.int32, device=device
        )
        cu_seqlens_q = torch.arange(
            capacity + 1, dtype=torch.int32, device=device
        )
        metadata = (lengths, cu_seqlens_q)
        capturing = (
            device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )
        if not capturing:
            _LIGHTOP_DCP_TOPK_METADATA[key] = metadata
    lengths, cu_seqlens_q = metadata
    return lengths[:rows], cu_seqlens_q[: rows + 1]


def get_decode_topk_output_buffer(
    topk_indices_buffer: torch.Tensor,
    num_padded_tokens: int,
    topk_tokens: int,
    requires_padding: bool,
) -> torch.Tensor:
    """Return storage for a decode Top-K kernel result.

    For a uniform/flattened decode batch, the number of kernel rows equals the
    number of actual decode rows and the shared prefix can be used directly.
    For a padded batch, allocate a disjoint temporary tensor.  The caller must
    compact that tensor with :func:`unpack_seq_triton` before copying the
    actual rows back to ``topk_indices_buffer``.

    ``topk_indices_buffer`` is deliberately not used as the padded scratch
    target.  Its rows after ``num_decode_tokens`` may contain prefill results.
    ``new_empty`` preserves the dtype and device while avoiding assumptions
    about the physical capacity or layout of the shared buffer.
    """
    if not requires_padding:
        return topk_indices_buffer[:num_padded_tokens, :topk_tokens]

    padded_topk = topk_indices_buffer.new_empty(
        (num_padded_tokens, topk_tokens)
    )
    # Top-K kernels generally write valid entries only.  Keep the same -1
    # sentinel contract as the shared buffer for invalid/padded entries.
    padded_topk.fill_(-1)
    return padded_topk


__all__ = [
    "get_decode_topk_output_buffer",
    "get_lightop_dcp_topk_metadata",
    "get_lightop_fast_topk_transform",
    "use_lightop_dcp_topk_transform",
]
