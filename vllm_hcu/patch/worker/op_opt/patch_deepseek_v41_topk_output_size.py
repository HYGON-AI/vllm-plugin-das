# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid scalar shape readback in the DSV4.1 AMD prefill index combiner."""

from __future__ import annotations

import functools
import os
from types import ModuleType

import torch

from ..core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
TARGETS = (f"{TARGET_MODULE}.combine_topk_swa_indices",)
PATCH_ID = "worker.op_opt.deepseek_v41.topk_output_size"
_MARKER = "_vllm_hcu_topk_output_size_applied"
_ORIGINAL = "_vllm_hcu_original_combine_topk_swa_indices"
_FLAG = "VLLM_HCU_DSV41_TOPK_OUTPUT_SIZE"
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the AMD Torch fallback without repeat-size scalar readback."""
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    # query_start_loc may have a non-zero base for a narrowed mixed batch.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens,
        output_size=num_tokens,
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens, output_size=num_tokens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        topk_values + (M * req_ids).to(torch.int32)[:, None],
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    swa_values = (
        M * req_ids[:, None]
        + N
        + swa_offsets[None, :]
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[req_ids, None]
    ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return combined_indices, combined_lens


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        if (
            not callable(getattr(target, _ORIGINAL, None))
            or not getattr(
                target.combine_topk_swa_indices,
                "_vllm_hcu_topk_output_size_wrapper",
                False,
            )
        ):
            raise PatchCompatibilityError(f"stale TopK patch on {TARGET_MODULE}")
        return False
    original = require_callable(
        target, "combine_topk_swa_indices", TARGETS[0]
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "topk_indices", "query_start_loc", "seq_lens", "gather_lens",
            "window_size", "compress_ratio", "topk", "M", "N",
        ),
    )
    if (
        getattr(target, "_SPARSE_PREFILL_TOPK_ALIGNMENT", None)
        != _SPARSE_PREFILL_TOPK_ALIGNMENT
    ):
        raise PatchCompatibilityError("unsupported AMD sparse-prefill alignment")

    @functools.wraps(original)
    def combine(
        topk_indices, query_start_loc, seq_lens, gather_lens,
        window_size, compress_ratio, topk, M, N,
    ):
        function = (
            _combine_topk_swa_indices
            if os.environ.get(_FLAG, "1").lower() in ("1", "true")
            else original
        )
        return function(
            topk_indices, query_start_loc, seq_lens, gather_lens,
            window_size, compress_ratio, topk, M, N,
        )

    combine._vllm_hcu_topk_output_size_wrapper = True
    setattr(target, _ORIGINAL, original)
    target.combine_topk_swa_indices = combine
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
