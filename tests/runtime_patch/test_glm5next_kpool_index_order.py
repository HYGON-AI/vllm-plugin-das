# SPDX-License-Identifier: Apache-2.0

from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_glm5next_channel_fp8 as patch
from vllm_hcu.platforms import hcu


def _boundary(source):
    """Replace only upstream GPU selection; exercise the real HCU wrapper."""
    module = ModuleType(patch.KPOOL_MODULE)

    def sparse_attn_indexer_kpool(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_quant,
        q_scale,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert,
        use_fp4_cache=False,
        gate_score=None,
        compress_ape=None,
        index_kpool=1,
        positions=None,
        tail_kv_cache=None,
        tail_prefix=None,
    ):
        # Mimic the upstream shared-buffer contract and only write active rows.
        rows = hidden_states.shape[0]
        topk_indices_buffer[:rows].copy_(source[:rows])
        return topk_indices_buffer

    module.sparse_attn_indexer_kpool = sparse_attn_indexer_kpool
    return module


def _call(module, hidden, result, *, pool=4, positional=False):
    args = (
        hidden,
        "indexer",
        None,
        None,
        None,
        None,
        None,
        128,
        None,
        result.shape[1],
        128,
        32768,
        32768,
        result,
        False,
    )
    if positional:
        return module.sparse_attn_indexer_kpool(
            *args, False, None, None, pool, None, None, None
        )
    return module.sparse_attn_indexer_kpool(*args, index_kpool=pool)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(hcu, "on_gfx938", lambda: True)


@pytest.mark.parametrize("positional", [False, True])
def test_ordering_preserves_alias_multiset_and_unused_rows(enabled, positional):
    # Removal of normalization, full-buffer sorting, replacement allocation,
    # dropping duplicates, or sorting -1 first each breaks this contract.
    source = torch.tensor([[7, -1, 3, 0, -1], [8, 2, 8, -1, 1]], dtype=torch.int32)
    storage = torch.full((4, 10), -9, dtype=torch.int32)
    result = storage[:, ::2]  # Genuine noncontiguous output buffer.
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    actual = _call(module, torch.zeros(2, 1), result, positional=positional)
    assert actual is result
    assert actual.data_ptr() == storage.data_ptr()
    assert actual.tolist() == [[0, 3, 7, -1, -1], [1, 2, 8, 8, -1], [-9] * 5, [-9] * 5]
    assert torch.all(storage[:, 1::2] == -9)
    for before, after in zip(source.tolist(), actual[:2].tolist()):
        assert sorted(before) == sorted(after)


@pytest.mark.parametrize("rows", [0, 1, 4, 50, 128])
def test_only_current_compute_rows_are_ordered(enabled, rows):
    source = torch.tensor([3, -1, 1, 0], dtype=torch.int32).repeat(128, 1)
    result = torch.full((256, 4), -9, dtype=torch.int32)
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    _call(module, torch.zeros(rows, 1), result)
    assert result[:rows].tolist() == [[0, 1, 3, -1]] * rows
    assert torch.all(result[rows:] == -9)


def test_all_padding_and_duplicate_indices_are_retained(enabled):
    source = torch.tensor([[-1, -1, -1, -1], [7, 7, 7, 7]], dtype=torch.int32)
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    result = torch.empty_like(source)
    _call(module, torch.zeros(2, 1), result)
    assert torch.equal(result, source)


@pytest.mark.parametrize("master,gfx938", [("0", True), ("1", False)])
def test_platform_and_master_guards_are_independent(monkeypatch, master, gfx938):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setattr(hcu, "on_gfx938", lambda: gfx938)
    source = torch.tensor([[3, -1, 1]], dtype=torch.int32)
    module = _boundary(source)
    original = module.sparse_attn_indexer_kpool
    assert not patch._patch_kpool_index_order(module)
    assert module.sparse_attn_indexer_kpool is original
    result = torch.empty_like(source)
    assert _call(module, torch.zeros(1, 1), result) is result
    assert torch.equal(result, source)


def test_master_defaults_to_enabled_without_child_environment(enabled, monkeypatch):
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS")
    source = torch.tensor([[3, -1, 1]], dtype=torch.int32)
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    result = torch.empty_like(source)
    _call(module, torch.zeros(1, 1), result)
    assert result.tolist() == [[1, 3, -1]]


def test_nonpooled_indexer_keeps_original_output(enabled):
    source = torch.tensor([[3, -1, 1]], dtype=torch.int32)
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    result = torch.empty_like(source)
    _call(module, torch.zeros(1, 1), result, pool=1)
    assert torch.equal(result, source)


def test_installation_is_idempotent_and_rejects_stale_marker(enabled):
    from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

    module = _boundary(torch.tensor([[3, 1]], dtype=torch.int32))
    original = module.sparse_attn_indexer_kpool
    assert patch._patch_kpool_index_order(module)
    wrapped = module.sparse_attn_indexer_kpool
    assert not patch._patch_kpool_index_order(module)
    assert module.sparse_attn_indexer_kpool is wrapped
    module.sparse_attn_indexer_kpool = original
    with pytest.raises(PatchCompatibilityError, match="marker is stale"):
        patch._patch_kpool_index_order(module)


def test_upstream_signature_mismatch_is_rejected(enabled):
    from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

    module = ModuleType(patch.KPOOL_MODULE)
    module.sparse_attn_indexer_kpool = lambda hidden_states: hidden_states
    with pytest.raises(PatchCompatibilityError):
        patch._patch_kpool_index_order(module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU graph replay")
def test_graph_replay_orders_new_permutations_in_place(enabled):
    rows, width, valid = 4, 2176, 2048
    expected = torch.cat(
        (
            torch.arange(valid, dtype=torch.int32),
            torch.full((width - valid,), -1, dtype=torch.int32),
        )
    )
    source = expected.roll(53).repeat(rows, 1).cuda()
    result = torch.full((128, width), -9, dtype=torch.int32, device="cuda")
    hidden = torch.zeros(rows, 1, device="cuda")
    module = _boundary(source)
    patch._patch_kpool_index_order(module)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            assert _call(module, hidden, result) is result
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _call(module, hidden, result)
    pointer = result.data_ptr()
    for shift in (11, 2170, 1):
        source.copy_(expected.roll(shift).repeat(rows, 1).cuda())
        graph.replay()
        torch.cuda.synchronize()
        assert actual is result and result.data_ptr() == pointer
        assert torch.equal(result[:rows].cpu(), expected.repeat(rows, 1))
        assert torch.all(result[rows:] == -9)
