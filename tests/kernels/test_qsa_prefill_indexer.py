# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical tests for the request-major QSA prefill scorer."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_hcu.v1.attention.ops.qsa_prefill_indexer import (
    _prefill_chunks,
    qsa_mqa_paged_prefill,
)


def _case(*, batch: int, query_len: int, prefix_len: int):
    from flash_attn import mqa_paged_score_func

    torch.manual_seed(248)
    device = "cuda"
    heads, head_dim = 4, 128
    compress_ratio, page_size = 4, 4
    sequence_len = prefix_len + query_len
    compressed_len = sequence_len // compress_ratio
    pages_per_request = math.ceil(compressed_len / page_size)
    total_pages = batch * pages_per_request

    q = torch.randn(
        batch * query_len,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    logical_cache = torch.randn(
        total_pages,
        page_size,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    permutation = torch.randperm(total_pages, device=device)
    k_cache = torch.empty_like(logical_cache)
    k_cache[permutation] = logical_cache
    page_table = permutation.reshape(batch, pages_per_request).to(torch.int32)

    query_start_loc = torch.arange(
        0,
        (batch + 1) * query_len,
        query_len,
        dtype=torch.int32,
        device=device,
    )
    token_to_req = torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(
        query_len
    )
    local_positions = torch.arange(
        prefix_len,
        sequence_len,
        dtype=torch.int64,
        device=device,
    )
    query_positions = local_positions.repeat(batch)
    sequence_lengths = torch.full(
        (batch,), sequence_len, dtype=torch.int32, device=device
    )
    expected, visible_blocks = mqa_paged_score_func(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    return q, k_cache, page_table, query_start_loc, visible_blocks, expected


def _assert_scores_and_topk(actual, expected, visible_blocks, *, row_offset=0):
    assert actual.dtype == torch.float32
    rows, columns = actual.shape
    column_ids = torch.arange(columns, device=actual.device)
    visible = visible_blocks[row_offset : row_offset + rows]
    valid = column_ids[None, :] < visible[:, None]
    torch.testing.assert_close(
        actual[valid],
        expected[row_offset : row_offset + rows][valid],
        rtol=2e-2,
        atol=2e-2,
    )
    actual_masked = actual.masked_fill(~valid, -torch.inf)
    expected_masked = expected[row_offset : row_offset + rows].masked_fill(
        ~valid, -torch.inf
    )
    for row in range(rows):
        k = min(16, int(visible[row]))
        if k:
            assert torch.equal(
                actual_masked[row].topk(k).indices.sort().values,
                expected_masked[row].topk(k).indices.sort().values,
            )


def test_prefill_chunks_bound_workspace_without_dropping_rows():
    assert _prefill_chunks(10, 4, 64) == ((0, 4), (4, 8), (8, 10))
    assert _prefill_chunks(0, 4, 64) == ()


def test_prefill_rejects_query_offsets_with_wrong_request_count():
    q = torch.empty(8, 4, 128, dtype=torch.bfloat16)
    cache = torch.empty(2, 4, 1, 128, dtype=torch.bfloat16)
    table = torch.zeros(2, 1, dtype=torch.int32)
    bad_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    visible = torch.zeros(8, dtype=torch.int32)

    with pytest.raises(ValueError, match="query_start_loc"):
        qsa_mqa_paged_prefill(
            q,
            cache,
            table,
            bad_start_loc,
            visible,
            4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize("prefix_len", [0, 128])
def test_prefill_scores_match_cutlass_with_permuted_pages(prefix_len):
    q, cache, table, starts, visible, expected = _case(
        batch=2, query_len=128, prefix_len=prefix_len
    )

    actual = qsa_mqa_paged_prefill(
        q,
        cache,
        table,
        starts,
        visible,
        128,
    )

    _assert_scores_and_topk(actual, expected, visible)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
def test_prefill_chunk_offset_keeps_absolute_request_rows():
    q, cache, table, starts, visible, expected = _case(
        batch=2, query_len=128, prefix_len=128
    )
    row_offset, num_rows = 96, 96

    actual = qsa_mqa_paged_prefill(
        q,
        cache,
        table,
        starts,
        visible,
        128,
        query_offset=row_offset,
        num_queries=num_rows,
    )

    _assert_scores_and_topk(
        actual,
        expected,
        visible,
        row_offset=row_offset,
    )

