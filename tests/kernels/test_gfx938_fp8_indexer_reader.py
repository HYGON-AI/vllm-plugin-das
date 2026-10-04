# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical reader tests with independently packed HIPC pages."""

import math

import pytest
import torch

from vllm_hcu.v1.attention.ops.fp8_paged_mqa_gfx938 import (
    gfx938_fp8_paged_mqa_logits,
)


def _case(
    page_size,
    length,
    batch,
    next_n,
    per_query,
    dtype=torch.float8_e4m3fn,
    heads=64,
    dim=128,
    metadata_dtype=torch.int32,
):
    torch.manual_seed(123)
    device = "cuda"
    pages = math.ceil(length / page_size)
    tokens = pages * page_size
    # Retain original token-order values for the oracle; never unpack the cache.
    keys = torch.randn(batch, tokens, dim, device=device).to(dtype)
    scales = torch.rand(batch, tokens, device=device) * 0.15 + 0.01
    q = torch.randn(batch, next_n, heads, dim, device=device).to(dtype)
    weights = torch.randn(batch * next_n, heads, device=device) * 0.05
    total_pages = batch * pages
    permutation = torch.randperm(total_pages, device=device)
    cache = torch.empty(
        total_pages + 1, page_size * (dim + 4), dtype=torch.uint8, device=device
    )
    # HIPC tiles are [token tile, dimension tile, token lane, dimension lane].
    packed = (
        keys.view(torch.uint8)
        .reshape(total_pages, page_size // 16, 16, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, page_size * dim)
    )
    cache[permutation, : page_size * dim] = packed
    cache[permutation, page_size * dim :] = scales.reshape(total_pages, page_size).view(
        torch.uint8
    )
    # A negative page index wrapping to the last page would read this sentinel.
    cache[-1, : page_size * dim] = (
        torch.full((page_size * dim,), 224.0, device=device).to(dtype).view(torch.uint8)
    )
    cache[-1, page_size * dim :] = torch.full((page_size,), 1024.0, device=device).view(
        torch.uint8
    )
    table = torch.full((batch, pages + 1), -1, dtype=metadata_dtype, device=device)
    table[:, :pages] = permutation.reshape(batch, pages).to(torch.int32)
    if length > page_size:
        table[0, 1] = -1
    final_lengths = length - torch.arange(batch, device=device) % 3
    ends = final_lengths[:, None] - next_n + 1 + torch.arange(next_n, device=device)
    context = ends.to(metadata_dtype) if per_query else final_lengths.to(metadata_dtype)
    max_len = tokens + page_size - 3
    expected = torch.full((batch, next_n, max_len), -torch.inf, device=device)
    positions = torch.arange(tokens, device=device)
    for b in range(batch):
        dots = q[b].float().reshape(next_n * heads, dim) @ keys[b].float().T
        scores = (
            dots.reshape(next_n, heads, tokens).relu()
            * weights.reshape(batch, next_n, heads)[b, :, :, None]
        ).sum(1) * scales[b]
        valid = positions[None, :] < ends[b, :, None]
        if b == 0 and length > page_size:
            valid &= (positions < page_size) | (positions >= 2 * page_size)
        expected[b, :, :tokens] = scores.masked_fill(~valid, -torch.inf)
    physical_cache = cache.view(total_pages + 1, page_size, 1, dim + 4)
    return (q, physical_cache, weights, context, table, max_len), expected.flatten(0, 1)


def _check(actual, expected):
    assert actual.dtype == torch.float32
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-2)
    max_error = (actual[finite] - expected[finite]).abs().max().item()
    checked = 0
    for row in range(expected.shape[0]):
        count = int(torch.isfinite(expected[row]).sum())
        k = min(32, count)
        if not k:
            continue
        values, indices = expected[row].topk(k)
        chosen = actual[row].topk(k).indices
        # Membership may differ only within the declared numerical tie band.
        tolerance = 0.01 + 0.001 * values[-1].abs()
        certain = indices[values > values[-1] + 2 * tolerance]
        assert torch.isin(certain, chosen).all()
        assert (expected[row, chosen] >= values[-1] - 2 * tolerance).all()
        checked += certain.numel()
    print(f"max_abs_error={max_error:.8g}, non_tied_topk_checked={checked}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("length", [2, 513])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("per_query", [False, True])
def test_direct_reader_matches_token_order_oracle(
    page_size, length, batch, next_n, per_query
):
    args, expected = _case(page_size, length, batch, next_n, per_query)
    _check(gfx938_fp8_paged_mqa_logits(*args), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize("per_query", [False, True])
def test_non_power_of_two_heads_and_dim_with_int64_metadata(per_query):
    """Tail lanes and 64-bit page indices must preserve oracle logits."""
    args, expected = _case(
        32, 37, 1, 3, per_query, heads=19, dim=48, metadata_dtype=torch.int64
    )
    _check(gfx938_fp8_paged_mqa_logits(*args), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize(
    "page_size,batch,next_n,per_query",
    [
        (16, 64, 1, False),
        (32, 1, 1, False),
        (32, 8, 3, True),
        (32, 64, 1, False),
        (32, 64, 3, True),
        (64, 64, 1, True),
    ],
)
def test_long_context_reader_matches_token_order_oracle(
    page_size, batch, next_n, per_query
):
    args, expected = _case(page_size, 32768, batch, next_n, per_query)
    _check(gfx938_fp8_paged_mqa_logits(*args), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize("length_exceeds_table", [False, True])
def test_short_block_table_masks_uncovered_output(length_exceeds_table):
    args, expected = _case(32, 513, 1, 1, False)
    q, cache, weights, lengths, table, max_len = args
    if length_exceeds_table:
        table = table[:, :4]
        expected[:, 128:] = -torch.inf
    else:
        table = table[:, :-1]
    actual = gfx938_fp8_paged_mqa_logits(q, cache, weights, lengths, table, max_len)
    _check(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
def test_graph_replay_reads_updated_lengths_and_pages():
    args, expected = _case(32, 513, 1, 3, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gfx938_fp8_paged_mqa_logits(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gfx938_fp8_paged_mqa_logits(*args)
    graph.replay()
    _check(actual, expected)
    args[3].fill_(100)
    args[4][0, 0] = -1
    expected[:, :32] = -torch.inf
    expected[:, 100:] = -torch.inf
    graph.replay()
    _check(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
def test_strided_inputs_match_token_order_oracle():
    args, expected = _case(32, 513, 1, 3, True)
    strided = []
    for index, tensor in enumerate(args[:-1]):
        if index == 1:
            strided.append(tensor.view(torch.float8_e4m3fn))
            continue
        axis = -2 if index == 0 else -1
        shape = list(tensor.shape)
        shape[axis] *= 2
        padded = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(None, None, 2)
        view = padded[tuple(slices)]
        view.copy_(tensor)
        strided.append(view)
    _check(gfx938_fp8_paged_mqa_logits(*strided, args[-1]), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
def test_layer_interleaved_physical_pages_match_oracle():
    args, expected = _case(32, 513, 1, 3, True)
    q, cache, weights, lengths, table, max_len = args
    page_size = cache.shape[1]
    backing = torch.empty(
        (cache.shape[0], page_size + 1, *cache.shape[2:]),
        dtype=cache.dtype,
        device=cache.device,
    )
    interleaved_pages = backing[:, :page_size]
    interleaved_pages.copy_(cache)
    assert not interleaved_pages.is_contiguous()
    assert interleaved_pages.stride(0) > cache.stride(0)

    actual = gfx938_fp8_paged_mqa_logits(
        q,
        interleaved_pages,
        weights,
        lengths,
        table,
        max_len,
    )

    _check(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
@pytest.mark.parametrize(
    "invalid", ["page_size", "cache_stride", "dtype", "fnuz", "table"]
)
def test_unsupported_physical_layout_rejected_before_launch(invalid):
    args, _ = _case(32, 2, 1, 1, False)
    args = list(args)
    if invalid == "page_size":
        args[1] = args[1].reshape(-1, 1, 1, 132)
    elif invalid == "cache_stride":
        args[1] = args[1][:, ::2]
    elif invalid == "dtype":
        args[0] = args[0].float()
    elif invalid == "fnuz":
        args[0] = args[0].view(torch.float8_e4m3fnuz)
    else:
        args[4] = args[4][:, :0]
    with pytest.raises(ValueError):
        gfx938_fp8_paged_mqa_logits(*args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU GPU")
def test_out_of_range_page_id_is_masked_before_cache_read():
    args, expected = _case(32, 513, 1, 1, False)
    args[4][0, 0] = args[1].shape[0] + 123
    expected[:, :32] = -torch.inf
    _check(gfx938_fp8_paged_mqa_logits(*args), expected)
