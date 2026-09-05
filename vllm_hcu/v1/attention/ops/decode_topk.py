# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Utilities for staging sparse-MLA decode Top-K results.

The sparse indexer stores decode and prefill results in one logical output
buffer.  A variable-length decode batch is temporarily expanded to a
rectangular ``[batch, max_decode_len]`` layout, so its Top-K kernel can produce
more rows than there are actual decode tokens.  The padded rows must never be
written into the shared output buffer: the rows immediately following the
decode prefix belong to prefill.
"""

from __future__ import annotations

import torch


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


__all__ = ["get_decode_topk_output_buffer"]
