# SPDX-License-Identifier: Apache-2.0
"""Stable attention order must preserve selection, padding and buffer tails."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4.sparse_indices import canonicalize_topk_indices


def _reference(indices):
    keys = torch.where(indices < 0, torch.iinfo(torch.int32).max, indices)
    ordered = keys.sort(dim=-1).values
    return torch.where(ordered == torch.iinfo(torch.int32).max, -1, ordered)


@pytest.mark.hcu
@pytest.mark.parametrize("width", [1, 7, 128, 2048])
@pytest.mark.parametrize("strided", [False, True])
def test_canonicalization_preserves_selection_and_inactive_rows(width, strided):
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    backing = torch.full((8, width * 2), -123, device="cuda", dtype=torch.int32)
    indices = backing[::2, ::2] if strided else backing[:, :width]
    generator = torch.Generator(device="cuda").manual_seed(338)
    values = torch.randint(
        -1, 8192, indices.shape, generator=generator, device="cuda", dtype=torch.int32
    )
    values[:, ::3] = -1
    indices.copy_(values)
    original = backing.clone()
    expected = values.clone()
    expected[:3] = _reference(values[:3])
    canonicalize_topk_indices(indices, 3)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    after = backing.clone()
    if strided:
        after[::2, ::2] = original[::2, ::2]
    else:
        after[:, :width] = original[:, :width]
    torch.testing.assert_close(after, original, rtol=0, atol=0)
    canonicalize_topk_indices(indices, 3)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.hcu
def test_graph_replay_reads_new_indices_and_preserves_sentinels():
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    indices = torch.full((4, 2048), -1, device="cuda", dtype=torch.int32)
    indices[1] = 12345
    canonicalize_topk_indices(indices, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        canonicalize_topk_indices(indices, 1)
    for length in [1, 17, 2048, 2039, 0]:
        row = torch.full((2048,), -1, device="cuda", dtype=torch.int32)
        row[:length] = torch.arange(length, 0, -1, device="cuda", dtype=torch.int32)
        indices[0].copy_(row)
        graph.replay()
        torch.testing.assert_close(indices[0], _reference(row), rtol=0, atol=0)
        assert (indices[1] == 12345).all().item()


def test_canonicalization_rejects_invalid_buffer_contract():
    with pytest.raises(ValueError, match="2D int32"):
        canonicalize_topk_indices(torch.zeros(2, 2), 1)
    with pytest.raises(ValueError, match="active"):
        canonicalize_topk_indices(torch.zeros(2, 2, dtype=torch.int32), 3)
    canonicalize_topk_indices(torch.empty(0, 2048, dtype=torch.int32), 0)


@pytest.mark.parametrize("enabled", [True, False])
def test_indexer_orders_only_active_rows_and_retains_shared_buffer(
    monkeypatch, enabled
):
    from vllm_hcu.models.hy_v4 import attention

    hidden = torch.zeros(1, 4)
    buffer = torch.zeros(1024, 8, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(
        attention,
        "canonicalize_topk_indices",
        lambda indices, count: calls.append((indices, count)),
    )
    indexer = SimpleNamespace(
        canonicalize_topk=enabled,
        prepare_inputs=lambda *args: (hidden, hidden, hidden, hidden),
        indexer_op=lambda *args: buffer,
    )
    result = attention.Indexer.forward(indexer, hidden, hidden, hidden, None)
    assert result is buffer
    if enabled:
        assert len(calls) == 1
        assert calls[0][0] is buffer
        assert calls[0][1] == 1
    else:
        assert calls == []
