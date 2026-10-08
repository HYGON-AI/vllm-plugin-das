# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""NORMAL sparse-indexer K-cache pages: official writers feed LightOp decode."""

import math
from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_glm5next_channel_fp8 as glm_patch
from vllm_hcu.platforms.hcu import on_gfx938

# The dispatcher rejects kcache_layout="normal" off gfx938.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not on_gfx938(),
    reason="requires gfx938 HCU",
)


def _pin_normal_layout(monkeypatch):
    from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    for module, kernels in (
        (kpool_ops, glm_patch._NORMAL_LAYOUT_KPOOL_KERNELS),
        (upstream_sparse, glm_patch._NORMAL_LAYOUT_SPARSE_KERNELS),
    ):
        for attr, _, _ in kernels:
            # Restore the official kernels after the test.
            monkeypatch.setattr(module, attr, getattr(module, attr))
    kpool = ModuleType(glm_patch.KPOOL_MODULE)
    kpool.kpool_ops = kpool_ops
    glm_patch._install_normal_indexer_kcache_layout(kpool, upstream_sparse)
    return kpool_ops, upstream_sparse


def _decode(q, cache, weights, lengths, table, max_len):
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

    return hcu_sparse.rocm_fp8_paged_mqa_logits(
        q, cache, weights, lengths, table, None, max_len, kcache_layout="normal"
    )


def _oracle(q, keys, scales, weights, lengths, max_len):
    batch, length, _ = keys.shape
    scores = torch.matmul(q[:, 0].float(), keys.float().transpose(1, 2))
    scores = (scores.relu() * weights[:, :, None]).sum(1) * scales
    expected = torch.full((batch, max_len), -torch.inf, device=q.device)
    expected[:, :length] = scores
    for b in range(batch):
        expected[b, int(lengths[b]) :] = -torch.inf
    return expected


def _check(actual, expected):
    actual = actual.reshape(expected.shape[0], -1)[:, : expected.shape[1]]
    finite = torch.isfinite(expected)
    assert torch.isneginf(actual[~finite]).all()
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-2)


def _paged(batch, length, page_size, dim, device):
    pages_per_row = length // page_size
    num_pages = batch * pages_per_row
    table = (
        torch.randperm(num_pages, device=device)
        .reshape(batch, pages_per_row)
        .to(torch.int32)
    )
    slots = torch.arange(length, device=device)
    locs = (
        table[:, slots // page_size].to(torch.int64) * page_size + slots % page_size
    ).flatten()
    cache = torch.zeros(
        (num_pages, page_size, 1, dim + 4), dtype=torch.uint8, device=device
    )
    return table, locs, cache


@pytest.mark.parametrize("page_size", [16, 32, 64])
def test_kpool_writer_normal_pages_feed_lightop(monkeypatch, page_size):
    kpool_ops, _ = _pin_normal_layout(monkeypatch)
    torch.manual_seed(3302)
    device = "cuda"
    batch, length, pool_size, heads, dim = 2, 1024, 4, 32, 128
    table, locs, cache = _paged(batch, length, page_size, dim, device)
    raw_keys = torch.randn(
        (batch * length, pool_size, dim), dtype=torch.bfloat16, device=device
    )
    gate_scores = (torch.randn(raw_keys.shape, device=device) * 0.05).to(
        torch.bfloat16
    )
    ape = torch.zeros((pool_size, dim), dtype=torch.float32, device=device)
    keys, scales = kpool_ops.kpool_compress_and_write_cache(
        cache, raw_keys, gate_scores, ape, locs, pool_size, return_compressed=True
    )
    # NORMAL pages hold each token's FP8 key contiguously.
    values = cache.reshape(cache.shape[0], -1)[:, : page_size * dim]
    token_major = values.view(torch.float8_e4m3fn).reshape(-1, dim)[locs]
    assert torch.equal(token_major.view(torch.uint8), keys.view(torch.uint8))

    q = torch.randn((batch, 1, heads, dim), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((batch, heads), device=device) * 0.05
    lengths = torch.tensor([length, length - 3], dtype=torch.int32, device=device)
    max_len = length + page_size
    actual = _decode(q, cache, weights, lengths, table, max_len)
    expected = _oracle(
        q,
        keys.reshape(batch, length, dim),
        scales.reshape(batch, length),
        weights,
        lengths,
        max_len,
    )
    _check(actual, expected)


@pytest.mark.parametrize("page_size", [16, 64])
def test_standard_writer_and_prefill_gather_round_trip_normal_pages(
    monkeypatch, page_size
):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    torch.manual_seed(3303)
    device = "cuda"
    batch, length, heads, dim = 2, 512, 32, 128
    k = torch.randn((batch * length, dim), dtype=torch.bfloat16, device=device)

    def write_and_gather():
        table, locs, cache = _paged(batch, length, page_size, dim, device)
        upstream_sparse.indexer_k_quant_and_cache_triton(
            k, cache.squeeze(2), locs, dim, "ue8m0"
        )
        k_fp8 = torch.empty((batch * length, dim), dtype=torch.float8_e4m3fn,
                            device=device)
        k_scale = torch.empty((batch * length, 4), dtype=torch.uint8, device=device)
        cu = torch.arange(0, (batch + 1) * length, length, dtype=torch.int32,
                          device=device)
        token_to_seq = torch.arange(batch, dtype=torch.int32,
                                    device=device).repeat_interleave(length)
        upstream_sparse.cp_gather_indexer_k_quant_cache_triton(
            cache.squeeze(2), k_fp8, k_scale, table, cu, token_to_seq=token_to_seq
        )
        return table, cache, k_fp8, k_scale.view(torch.float32).flatten()

    _, shuffled, official_k, official_scale = write_and_gather()
    _pin_normal_layout(monkeypatch)
    table, cache, normal_k, normal_scale = write_and_gather()
    # Same quantized keys after the round trip, but a different page layout.
    assert torch.equal(normal_k.view(torch.uint8), official_k.view(torch.uint8))
    assert torch.equal(normal_scale, official_scale)
    assert not torch.equal(cache, shuffled)

    q = torch.randn((batch, 1, heads, dim), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((batch, heads), device=device) * 0.05
    lengths = torch.tensor([length, length - 5], dtype=torch.int32, device=device)
    max_len = length + page_size
    actual = _decode(q, cache, weights, lengths, table, max_len)
    expected = _oracle(
        q,
        normal_k.reshape(batch, length, dim),
        normal_scale.reshape(batch, length),
        weights,
        lengths,
        max_len,
    )
    _check(actual, expected)


def test_lightop_reads_layer_interleaved_normal_pages(monkeypatch):
    kpool_ops, _ = _pin_normal_layout(monkeypatch)
    torch.manual_seed(3304)
    device = "cuda"
    batch, length, pool_size, heads, dim, page_size = 2, 512, 4, 32, 128, 32
    table, locs, cache = _paged(batch, length, page_size, dim, device)
    backing = torch.zeros(
        (cache.shape[0], page_size + 1, 1, dim + 4), dtype=torch.uint8, device=device
    )
    raw_keys = torch.randn(
        (batch * length, pool_size, dim), dtype=torch.bfloat16, device=device
    )
    gate_scores = torch.zeros_like(raw_keys)
    ape = torch.zeros((pool_size, dim), dtype=torch.float32, device=device)
    keys, scales = kpool_ops.kpool_compress_and_write_cache(
        cache, raw_keys, gate_scores, ape, locs, pool_size, return_compressed=True
    )
    interleaved = backing[:, :page_size]
    interleaved.copy_(cache)
    assert interleaved.stride(0) > cache.stride(0)

    q = torch.randn((batch, 1, heads, dim), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((batch, heads), device=device) * 0.05
    lengths = torch.tensor([length, length - 7], dtype=torch.int32, device=device)
    max_len = length + page_size
    actual = _decode(q, interleaved, weights, lengths, table, max_len)
    expected = _oracle(
        q,
        keys.reshape(batch, length, dim),
        scales.reshape(batch, length),
        weights,
        lengths,
        max_len,
    )
    _check(actual, expected)


def _token_major_case(page_size, ends, per_query, heads=64, dim=128):
    """Token-major pages, mixed lengths, and a padded block table.

    ``ends[b, t]`` is the visible length of query token ``t``; 1D context
    lengths give the final length and LightOp derives the earlier tokens'.
    """
    torch.manual_seed(3305)
    device = "cuda"
    batch, next_n = ends.shape
    pages = max(1, math.ceil(int(ends.max()) / page_size))
    tokens = pages * page_size
    keys = torch.randn(batch, tokens, dim, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(batch, tokens, device=device) * 0.15 + 0.01
    q = torch.randn(batch, next_n, heads, dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch * next_n, heads, device=device) * 0.05
    total_pages = batch * pages
    permutation = torch.randperm(total_pages, device=device)
    cache = torch.zeros(
        total_pages, page_size * (dim + 4), dtype=torch.uint8, device=device
    )
    cache[permutation, : page_size * dim] = keys.view(torch.uint8).reshape(
        total_pages, page_size * dim
    )
    cache[permutation, page_size * dim :] = scales.reshape(total_pages, page_size).view(
        torch.uint8
    )
    # A trailing unused column, as in padded decode block tables.
    table = torch.zeros((batch, pages + 1), dtype=torch.int32, device=device)
    table[:, :pages] = permutation.reshape(batch, pages).to(torch.int32)
    context = ends if per_query else ends[:, -1].contiguous()
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
        expected[b, :, :tokens] = scores.masked_fill(~valid, -torch.inf)
    physical = cache.view(total_pages, page_size, 1, dim + 4)
    return (q, physical, weights, context, table, max_len), expected.flatten(0, 1)


def _check_rows(actual, expected, topk_k=32):
    """Logits, masked positions, and non-tied top-k membership per row."""
    actual = actual.reshape(expected.shape[0], -1)[:, : expected.shape[1]]
    finite = torch.isfinite(expected)
    assert torch.isneginf(actual[~finite]).all()
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-2)
    for row in range(expected.shape[0]):
        k = min(topk_k, int(finite[row].sum()))
        if not k:
            continue
        values, indices = expected[row].topk(k)
        chosen = actual[row].topk(k).indices
        tolerance = 0.01 + 0.001 * values[-1].abs()
        assert torch.isin(indices[values > values[-1] + 2 * tolerance], chosen).all()


def _causal_ends(final_lengths, next_n):
    final = torch.tensor(final_lengths, dtype=torch.int32, device="cuda")
    offsets = torch.arange(next_n, dtype=torch.int32, device="cuda")
    return final[:, None] - next_n + 1 + offsets


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("per_query", [False, True])
def test_lightop_normal_pages_mtp_verify_batches(page_size, next_n, per_query):
    # Mixed final lengths, including two requests with the same length.
    ends = _causal_ends([513, 512, 511, 513, 37, 129], next_n)
    args, expected = _token_major_case(page_size, ends, per_query)
    _check_rows(_decode(*args), expected)


@pytest.mark.parametrize("page_size", [16, 64])
def test_lightop_normal_pages_repeated_pool_lengths_and_padding_rows(page_size):
    # Pool-granular lengths of MTP3 verify tokens: a pool completes on no
    # token, the last token, or the first token; zero rows are padding.
    ends = torch.tensor(
        [[300, 300, 300], [300, 300, 301], [301, 301, 301], [0, 0, 0],
         [1, 1, 2], [0, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    args, expected = _token_major_case(page_size, ends, per_query=True)
    actual = _decode(*args).reshape(expected.shape[0], -1)
    _check_rows(actual, expected)
    padding = (ends.flatten() == 0).nonzero().flatten()
    assert torch.isneginf(actual[padding, : expected.shape[1]]).all()


def _unshuffle(cache, page_size, dim):
    # Inverse of the 16x16 tiles: [token tile, dim tile, token lane, dim lane].
    pages = cache.shape[0]
    keys = cache.reshape(pages, -1)[:, : page_size * dim]
    keys = keys.reshape(pages, page_size // 16, dim // 16, 16, 16)
    return keys.permute(0, 1, 3, 2, 4).reshape(pages, page_size * dim)


@pytest.mark.parametrize("page_size", [16, 64])
def test_batched_decode_update_writer_feeds_lightop(monkeypatch, page_size):
    from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops

    torch.manual_seed(3306)
    device = "cuda"
    pool_size, next_n, heads, dim = 4, 3, 32, 128
    # Pools already in the cache before this MTP3 verify step; two requests
    # share a length. The phase decides which token, if any, completes a pool.
    prefix = [37, 37, 64, 20]
    phase = [1, 3, 0, 0]
    real = [True, True, True, False]
    batch = len(prefix)
    pages = math.ceil((max(prefix) + 1) / page_size)
    num_pages = batch * pages + 1
    table = (
        torch.randperm(num_pages - 1, device=device)
        .reshape(batch, pages)
        .to(torch.int32)
    )
    cache = torch.zeros(
        (num_pages, page_size, dim + 4), dtype=torch.uint8, device=device
    )
    flat = cache.view(num_pages, -1)
    flat[:, : page_size * dim] = (
        torch.randn(num_pages, page_size * dim, device=device)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    flat[:, page_size * dim :] = (
        torch.rand(num_pages, page_size, device=device) * 0.15 + 0.01
    ).view(torch.uint8)
    shuffled = cache.clone()
    shuffled.view(num_pages, -1)[:, : page_size * dim] = (
        cache.view(num_pages, -1)[:, : page_size * dim]
        .reshape(num_pages, page_size // 16, 16, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(num_pages, page_size * dim)
    )
    # Random earlier stashes in each request's tail ring.
    tail = torch.randn(
        (batch, 2, pool_size, dim), dtype=torch.bfloat16, device=device
    )
    tail[:, 1] *= 0.5
    key = torch.randn((batch, next_n, dim), dtype=torch.bfloat16, device=device)
    score = (torch.randn(key.shape, device=device) * 0.5).to(torch.bfloat16)
    ape = torch.randn((pool_size, dim), dtype=torch.float32, device=device) * 0.1

    positions = torch.full((batch, next_n), -1, dtype=torch.int32, device=device)
    slots = torch.full_like(positions, -1)
    tail_slots = torch.full_like(positions, -1)
    ends = torch.zeros_like(positions)
    for b in range(batch):
        if not real[b]:
            continue
        for t in range(next_n):
            pos = prefix[b] * pool_size + phase[b] + t
            pool = pos // pool_size
            positions[b, t] = pos
            slots[b, t] = table[b, pool // page_size] * page_size + pool % page_size
            tail_slots[b, t] = b * pool_size + pos % pool_size
            ends[b, t] = (pos + 1) // pool_size

    def write(kv_cache):
        kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
            kv_cache, tail.clone(), tail_slots, key, score, ape, slots,
            positions, pool_size, dim,
        )

    write(shuffled)
    _pin_normal_layout(monkeypatch)
    before = cache.clone()
    write(cache)

    # Exactly the completed pools were written, identically in both layouts.
    def token_bytes(kv_cache):
        flat = kv_cache.view(num_pages, -1)
        keys = flat[:, : page_size * dim].reshape(num_pages, page_size, dim)
        scales = flat[:, page_size * dim :].reshape(num_pages, page_size, 4)
        return torch.cat([keys, scales], dim=-1)

    completed = ends[:, -1] > torch.tensor(prefix, device=device)
    written = (token_bytes(cache) != token_bytes(before)).any(-1)
    written = written.nonzero().tolist()
    expected_locs = sorted(
        divmod(int(table[b, prefix[b] // page_size]) * page_size
               + prefix[b] % page_size, page_size)
        for b in range(batch) if completed[b]
    )
    assert sorted(map(tuple, written)) == expected_locs
    assert torch.equal(
        _unshuffle(shuffled, page_size, dim),
        cache.view(num_pages, -1)[:, : page_size * dim],
    )
    assert torch.equal(
        shuffled.view(num_pages, -1)[:, page_size * dim :],
        cache.view(num_pages, -1)[:, page_size * dim :],
    )

    q = torch.randn((batch, next_n, heads, dim), device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn((batch * next_n, heads), device=device) * 0.05
    max_len = pages * page_size + 5
    tokens = pages * page_size
    pool_keys = cache.view(num_pages, -1)[:, : page_size * dim].view(
        torch.float8_e4m3fn
    ).reshape(num_pages, page_size, dim)
    pool_scales = cache.view(num_pages, -1)[:, page_size * dim :].view(
        torch.float32
    )
    expected = torch.full((batch, next_n, max_len), -torch.inf, device=device)
    for b in range(batch):
        k_b = pool_keys[table[b].long()].reshape(tokens, dim).float()
        s_b = pool_scales[table[b].long()].reshape(tokens)
        dots = q[b].float().reshape(next_n * heads, dim) @ k_b.T
        scores = (
            dots.reshape(next_n, heads, tokens).relu()
            * weights.reshape(batch, next_n, heads)[b, :, :, None]
        ).sum(1) * s_b
        valid = torch.arange(tokens, device=device)[None, :] < ends[b, :, None]
        expected[b, :, :tokens] = scores.masked_fill(~valid, -torch.inf)
    actual = _decode(
        q, cache.unsqueeze(2), weights, ends, table, max_len
    )
    _check_rows(actual, expected.flatten(0, 1))


def test_lightop_normal_mtp_fold_replays_in_cuda_graph():
    ends = _causal_ends([513, 300, 37, 1], 3)
    (q, cache, weights, context, table, max_len), _ = _token_major_case(
        64, ends, per_query=True
    )
    stream = torch.cuda.Stream()
    # Warm up on a side stream only after the inputs are written.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _decode(q, cache, weights, context, table, max_len)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _decode(q, cache, weights, context, table, max_len)
    # Replay with new per-query lengths, including a padding row.
    context.copy_(torch.tensor(
        [[400, 400, 401], [299, 300, 300], [0, 0, 0], [2, 2, 2]],
        dtype=torch.int32, device="cuda",
    ))
    graph.replay()
    torch.cuda.synchronize()
    eager = _decode(q, cache, weights, context, table, max_len)
    assert torch.equal(torch.isneginf(out), torch.isneginf(eager))
    finite = torch.isfinite(eager)
    torch.testing.assert_close(out[finite], eager[finite])
