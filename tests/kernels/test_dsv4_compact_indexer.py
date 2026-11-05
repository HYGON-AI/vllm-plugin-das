# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real LightOp decode from staged V4 pages, including graph buffer reuse."""

import pytest
import torch
from tests.kernels.test_gfx938_indexer_normal_layout import (
    _token_major_case, _causal_ends,
)
from vllm_hcu.platforms.hcu import on_gfx938
from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_paged_mqa_logits

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not on_gfx938(), reason="requires gfx938",
    ),
]


def _decode(args):
    q, cache, weights, lengths, table, max_len = args
    return rocm_fp8_paged_mqa_logits(
        q, cache, weights, lengths, table, None, max_len,
        kcache_layout="normal", prefer_direct=True, clean_logits=False,
    )


def _check(actual, expected, ends):
    valid = torch.arange(actual.shape[-1], device=actual.device)[None] < ends.reshape(-1, 1)
    assert torch.equal(torch.isneginf(actual[valid]), torch.isneginf(expected[valid]))
    finite = valid & torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("per_query", [False, True])
@pytest.mark.parametrize("fault", [None, "missing", "short"])
def test_staged_token_major_logits_match_oracle(flatten, per_query, fault):
    ends = (_causal_ends([513, 514], 6) if not per_query else
            torch.tensor([[513] * 6, [513, 513, 513, 514, 514, 514]],
                         device="cuda", dtype=torch.int32))
    args, expected = _token_major_case(64, ends, per_query)
    q, cache, w, lengths, table, max_len = args
    if fault == "missing":
        table[0, 1] = -1
        table[1, 2] = cache.shape[0] + 3
        expected[:6, 64:128] = -torch.inf
        expected[6:, 128:192] = -torch.inf
    elif fault == "short":
        table = table[:, :2]
        expected[:, 128:] = -torch.inf
    if flatten:
        q = q.reshape(-1, 1, 64, 128)
        lengths = ends.reshape(-1, 1)
        table = table.repeat_interleave(6, 0)
    _check(_decode((q, cache, w, lengths, table, max_len)), expected, ends)


def test_staged_graph_replay_refreshes_pages_lengths_and_invalid_entries():
    ends = torch.tensor([[513, 513, 513], [513, 513, 513]], device="cuda", dtype=torch.int32)
    args, expected = _token_major_case(64, ends, True)
    q, cache, w, lengths, table, max_len = args
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _decode(args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _decode(args)
    graph.replay()
    _check(actual, expected, ends)
    # A larger batch on the same stream must not invalidate older graphs.
    with torch.cuda.stream(stream):
        _decode((q.repeat(4, 1, 1, 1), cache, w.repeat(4, 1),
                 lengths.repeat(4, 1), table.repeat(4, 1), max_len))
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    _check(actual, expected, ends)
    # The new call shares all pages; no per-layer staging data may be stale.
    q[1].copy_(q[0])
    w[3:].copy_(w[:3])
    table[1].copy_(table[0])
    expected[3:] = expected[:3]
    table[:, 1] = -1
    expected[:, 64:128] = -torch.inf
    graph.replay()
    _check(actual, expected, ends)
    # An empty first query must not suppress the later queries' fresh copies.
    lengths[0, 0] = 0
    cache.zero_()
    expected = torch.where(torch.isfinite(expected), 0.0, -torch.inf)
    graph.replay()
    _check(actual, expected, lengths)


def test_staged_reader_handles_high_physical_page_addresses():
    ends = _causal_ends([513, 514], 6)
    args, expected = _token_major_case(64, ends, True)
    q, cache, w, lengths, table, max_len = args
    first, stride = 4000, 554496
    backing = torch.empty((first + cache.shape[0]) * stride, device="cuda", dtype=torch.uint8)
    physical = torch.as_strided(backing, (first + cache.shape[0], 64, 1, 132), (stride, 132, 132, 1))
    for page in range(cache.shape[0]):
        physical[first + page].copy_(cache[page])
    table = table + first
    _check(_decode((q, physical, w, lengths, table, max_len)), expected, ends)


def test_disabled_staging_keeps_direct_reader(monkeypatch):
    from vllm_hcu.platforms import envs
    from vllm_hcu.v1.attention.ops import dsv4_compact_indexer
    monkeypatch.setattr(envs, "VLLM_HCU_DSV4_INDEXER_COMPACT_DECODE", False)
    monkeypatch.setattr(dsv4_compact_indexer, "compact_indexer_logits",
                        lambda * a, **k:pytest.fail("staging must be disabled"))
    ends = _causal_ends([513, 514], 6)
    args, expected = _token_major_case(64, ends, True)
    _check(_decode(args), expected, ends)
