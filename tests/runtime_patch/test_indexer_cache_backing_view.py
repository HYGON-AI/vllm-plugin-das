# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for sparse-indexer KV-cache backing views."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse
from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    _indexer_cache_as_bf16_nhd_view,
    _indexer_cache_as_hipc_view,
)


class _FakeTritonKernel:
    def __init__(self):
        self.grid = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args):
            self.args = args

        return launch


def test_collapsed_indexer_cache_restores_physical_page_axis():
    backing = torch.empty(3, 64, 132, dtype=torch.uint8)
    collapsed = backing.as_strided(
        size=(3, 1, 132),
        stride=(64 * 132, 132, 1),
    )

    restored = _indexer_cache_as_hipc_view(collapsed)

    assert restored.shape == (3, 64, 132)
    assert restored.stride() == (64 * 132, 132, 1)
    assert restored.data_ptr() == collapsed.data_ptr()
    assert collapsed.shape == (3, 1, 132)


def test_regular_indexer_cache_is_not_reinterpreted():
    cache = torch.empty(3, 64, 132, dtype=torch.uint8)

    assert _indexer_cache_as_hipc_view(cache) is cache


def test_noncollapsed_padded_indexer_cache_is_not_reinterpreted():
    backing = torch.empty(3, 33, 132, dtype=torch.uint8)
    padded = backing.as_strided(
        size=(3, 32, 132),
        stride=(33 * 132, 132, 1),
    )

    assert _indexer_cache_as_hipc_view(padded) is padded


def test_logical_lbhnc_indexer_cache_has_zero_copy_bf16_nhd_view():
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    normalized = _indexer_cache_as_bf16_nhd_view(cache, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == cache.data_ptr()


def test_logical_lbnhc_indexer_cache_has_zero_copy_bf16_nhd_view():
    backing = torch.empty(3 * 64 * 128, dtype=torch.bfloat16)
    cache = backing.as_strided(
        size=(3, 1, 64, 128),
        stride=(64 * 128, 128, 128, 1),
    )

    normalized = _indexer_cache_as_bf16_nhd_view(cache, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == cache.data_ptr()


def test_collapsed_bf16_indexer_cache_restores_page_before_normalizing():
    backing = torch.empty(3, 64, 128, dtype=torch.bfloat16)
    collapsed = backing.as_strided(
        size=(3, 1, 128),
        stride=(64 * 128, 128, 1),
    )

    normalized = _indexer_cache_as_bf16_nhd_view(collapsed, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == collapsed.data_ptr()


def test_bf16_writer_passes_logical_4d_cache_as_nhd_to_kernel(monkeypatch):
    kernel = _FakeTritonKernel()
    monkeypatch.setattr(sparse, "_indexer_k_bf16_cache_kernel", kernel)
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    sparse.indexer_k_bf16_cache_triton(
        torch.empty(2, 128, dtype=torch.bfloat16),
        cache,
        torch.tensor([0, 65], dtype=torch.int32),
    )

    assert kernel.grid == (2,)
    assert kernel.args[1].shape == (3, 64 * 128)
    assert kernel.args[1].stride(0) == 64 * 128
    assert kernel.args[4:7] == (64, 2, 128)


def test_bf16_gather_passes_logical_4d_cache_as_nhd_to_kernel(monkeypatch):
    kernel = _FakeTritonKernel()
    monkeypatch.setattr(
        sparse, "_cp_gather_indexer_k_bf16_cache_kernel", kernel
    )
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    sparse.cp_gather_indexer_k_bf16_cache_triton(
        cache,
        torch.empty(2, 128, dtype=torch.bfloat16),
        torch.zeros(1, 2, dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert kernel.grid == (2,)
    assert kernel.args[0].shape == (3, 64 * 128)
    assert kernel.args[0].stride(0) == 64 * 128
    assert kernel.args[4:10] == (64, 1, 2, 64 * 128, 128, 2)


def test_bf16_gather_preserves_strided_block_table(monkeypatch):
    kernel = _FakeTritonKernel()
    monkeypatch.setattr(sparse, "_cp_gather_indexer_k_bf16_cache_kernel", kernel)
    table = torch.zeros(4, 8, dtype=torch.int32)[1::2, ::2]
    sparse.cp_gather_indexer_k_bf16_cache_triton(
        torch.empty(8, 64, 128, dtype=torch.bfloat16),
        torch.empty(128, 128, dtype=torch.bfloat16),
        table,
        torch.tensor([0, 64, 128], dtype=torch.int32),
    )
    assert kernel.args[2] is table
    assert kernel.args[-2:] == table.stride()


@pytest.mark.parametrize("gfx938", [False, True])
def test_native_decode_matches_custom_op_writer_layout(monkeypatch, gfx938):
    from vllm import _custom_ops as ops

    prefix = "test.indexer"
    decode = SimpleNamespace(
        decode_lens=torch.tensor([1], dtype=torch.int32),
        requires_padding=False,
        seq_lens=torch.tensor([[4]], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        schedule_metadata=None,
    )
    metadata = SimpleNamespace(
        num_kv_actual_tokens=1,
        slot_mapping=torch.tensor([3], dtype=torch.int64),
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        decode=decode,
    )
    monkeypatch.setattr(sparse, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(
        sparse, "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={prefix: metadata}),
    )
    monkeypatch.setattr(sparse, "on_gfx938", lambda: gfx938)
    monkeypatch.setattr(sparse, "_use_lightop_sparse_mla_topk", lambda: False)
    monkeypatch.setattr(ops, "indexer_k_quant_and_cache", lambda *args: None)
    monkeypatch.setattr(sparse, "indexer_k_bf16_cache_triton", lambda *args: None)
    calls = []

    def logits(*args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[1.0, 3.0, 2.0, 0.0, -torch.inf, -torch.inf]])

    monkeypatch.setattr(sparse, "rocm_fp8_paged_mqa_logits", logits)
    output = torch.empty((1, 2), dtype=torch.int32)
    result = sparse.rocm_aiter_sparse_attn_indexer_native(
        torch.zeros((1, 8)), prefix,
        torch.zeros((1, 64, 132), dtype=torch.uint8),
        torch.zeros((1, 2, 128), dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        torch.zeros((1, 128), dtype=torch.bfloat16),
        torch.ones((1, 2), dtype=torch.float32),
        128, "ue8m0", 2, 128, 6, None, output,
    )
    assert calls == [{
        "max_model_len": 6,
        "clean_logits": False,
        "kcache_layout": "normal" if gfx938 else "preshuffle",
    }]
    assert set(result[0].tolist()) == {1, 2}


@pytest.mark.parametrize("topk_tokens", [2, 2048])
def test_decode_topk_selects_required_ragged_api(monkeypatch, topk_tokens):
    calls = []
    logits = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    seq_lens = torch.tensor([3, 2], dtype=torch.int64)
    output = torch.empty((4, topk_tokens), dtype=torch.int32)
    expected_ends = torch.tensor([2, 3, 1, 2], dtype=torch.int32)

    def legacy(values, next_n, ends, result, *args):
        assert values is logits
        assert next_n == 1
        torch.testing.assert_close(ends, expected_ends)
        result.fill_(7)
        calls.append("legacy")

    def ragged(values, ends, starts, count, *, row_starts):
        assert values.is_contiguous()
        torch.testing.assert_close(values, logits)
        torch.testing.assert_close(ends, expected_ends)
        assert ends.is_contiguous()
        assert starts is row_starts
        torch.testing.assert_close(starts, torch.zeros_like(expected_ends))
        assert count == 2048
        calls.append("ragged")
        return torch.full_like(output, 9)

    attention = SimpleNamespace(
        top_k_per_row_decode=legacy,
        fast_topk_transform_ragged_fused=ragged,
    )
    monkeypatch.setattr(sparse, "_get_lightop_attention", lambda: attention)

    sparse._lightop_topk_indices_decode(logits, seq_lens, 2, output, topk_tokens)

    uses_ragged = topk_tokens == 2048
    assert calls == ["ragged" if uses_ragged else "legacy"]
    assert torch.all(output == (9 if uses_ragged else 7))


@pytest.mark.parametrize("use_lightop", [False, True])
@pytest.mark.parametrize("topk_tokens", [2, 2048])
@pytest.mark.parametrize("has_output", [False, True])
def test_indexer_profile_reserves_ragged_topk_output(
    monkeypatch, use_lightop, topk_tokens, has_output,
):
    hidden_states = torch.zeros((4, 8))
    output = torch.zeros((4, topk_tokens), dtype=torch.int32) if has_output else None
    allocations = []
    original_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch, "empty", tracked_empty)
    monkeypatch.setattr(sparse, "_use_lightop_sparse_mla_topk", lambda: use_lightop)
    result = sparse.rocm_aiter_sparse_attn_indexer_fake(
        hidden_states, "test.indexer", None, None, None, None,
        128, "ue8m0", topk_tokens, 128, 8192, 8, output,
    )

    assert result is output
    scratch = [buffer for buffer in allocations if buffer.shape == (4, topk_tokens)]
    assert len(scratch) == (2 if use_lightop and topk_tokens == 2048 else 1)
    assert all(buffer.dtype == torch.int32 for buffer in scratch)
    assert all(buffer.device == hidden_states.device for buffer in scratch)
    assert len({buffer.data_ptr() for buffer in scratch}) == len(scratch)
