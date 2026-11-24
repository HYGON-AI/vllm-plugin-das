# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PCP segment coalescing: native metadata and CPU tensor mapping oracle."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_transfers_do_not_require_a_gpu(monkeypatch):
    from vllm.utils import torch_utils
    monkeypatch.setattr(torch_utils, 'PIN_MEMORY', False)


def case(ids=(7, 7), lengths=(79354, 84835), queries=(189, 189), capacity=440000):
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadataBuilder
    from hcu_vllm_das.hy4_constructor import bind_constructor
    n = len(ids)
    cu = torch.tensor([0, *np.cumsum(queries).tolist()], dtype=torch.int32)
    # Every logical request has its own pages; its PCP segments share them.
    table = torch.tensor([[i * 1000 + j for j in range(1400)] for i in ids], dtype=torch.int32)
    common = NS(req_idx=np.array(ids), seq_lens_cpu_upper_bound=torch.tensor(lengths),
                seq_lens=torch.tensor(lengths, dtype=torch.int32), query_start_loc_cpu=cu,
                block_table_tensor=table, num_actual_tokens=int(cu[-1]))
    builder = NS(use_pcp=True, pcp_dcp_kv_gather=False, use_hisparse=False,
                 device=torch.device('cpu'), vllm_config=NS(model_config=NS(max_model_len=capacity),
                 attention_config=NS(sparse_mla_prefill_workspace_factor=1)), _hcu_workspace_l2_bytes=1)
    classification = NS(num_decodes=0, num_prefills=n, num_decode_tokens=0)
    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def zeros(self, *args, **kwargs):
            # Only the native builder's pinned allocation needs a CPU-only
            # replacement. Its body and all tensor operations remain native.
            kwargs.pop('pin_memory', None)
            return torch.zeros(*args, **kwargs)

    native_cpu = bind_constructor(FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode,
                                  {'torch': CpuTorch()})
    extra = native_cpu(builder, common, classification)
    return builder, common, extra


def coalesce(*args, **kwargs):
    from hcu_vllm_das.hy4_prefill_workspace import coalesce_prefill
    return coalesce_prefill(*args, **kwargs)


def test_two_pcp_segments_read_one_longest_context():
    builder, common, original = case()
    result = coalesce(builder, common, original)
    assert result is not original
    assert [int(c.chunk_tot_seqlen) for c in original.prefill.chunks] == [164189]
    chunk, = result.prefill.chunks
    assert chunk.chunk_tot_seqlen == 84835
    assert chunk.tokens_slice == slice(0, 378)
    assert chunk.seq_lens.tolist() == [84835]
    assert torch.equal(chunk.block_table[0], common.block_table_tensor[1])
    assert result.prefill.workspace_starts.tolist() == [0]
    assert result.prefill.request_ids.tolist() == [0] * 378
    assert result.num_prefill_tokens == 378


def test_distinct_requests_chunk_by_real_context_capacity():
    builder, common, original = case((7, 7, 9, 9, 11), (60, 85, 64, 90, 71),
                                     (2, 3, 1, 4, 2), capacity=180)
    result = coalesce(builder, common, original)
    assert [c.chunk_tot_seqlen for c in result.prefill.chunks] == [175, 71]
    assert [c.tokens_slice for c in result.prefill.chunks] == [slice(0, 10), slice(10, 12)]
    assert result.prefill.workspace_starts.tolist() == [0, 85, 0]
    assert result.prefill.request_ids.tolist() == [0] * 5 + [1] * 5 + [2] * 2
    # Scalar oracle: every local selected token still denotes the same page.
    for token, group in enumerate(result.prefill.request_ids.tolist()):
        row = next(i for i in range(5) if int(common.query_start_loc_cpu[i + 1]) > token)
        chunk = next(c for c in result.prefill.chunks if c.tokens_slice.start <= token < c.tokens_slice.stop)
        representative = group - chunk.req_start_idx
        for logical in (0, 17, min(int(common.seq_lens[row]) - 1, 70)):
            assert chunk.block_table[representative, logical // 64] == common.block_table_tensor[row, logical // 64]


@pytest.mark.parametrize('mode', ['no_pcp', 'dcp', 'hisparse', 'missing_ids', 'unique_ids', 'nonadjacent'])
def test_other_layouts_keep_the_exact_original(mode):
    builder, common, original = case()
    if mode == 'no_pcp': builder.use_pcp = False
    if mode == 'dcp': builder.pcp_dcp_kv_gather = True
    if mode == 'hisparse': builder.use_hisparse = True
    if mode == 'missing_ids': common.req_idx = None
    if mode == 'unique_ids': common.req_idx = np.array([7, 9])
    if mode == 'nonadjacent': builder, common, original = case((7, 9, 7), (20, 30, 40), (1, 2, 3))
    assert coalesce(builder, common, original) is original


def test_zero_query_segment_and_later_shorter_context():
    builder, common, original = case((7, 7), (85, 60), (0, 4), capacity=90)
    result = coalesce(builder, common, original)
    chunk, = result.prefill.chunks
    assert chunk.chunk_tot_seqlen == 85 and chunk.tokens_slice == slice(0, 4)
    assert torch.equal(chunk.block_table[0], common.block_table_tensor[0])


def test_empty_request_run_retains_native_metadata():
    builder, common, original = case((7, 7, 9, 9), (60, 85, 64, 90), (0, 0, 1, 2), capacity=90)
    assert coalesce(builder, common, original) is original


@pytest.mark.parametrize('cache_bytes', [0, 8 * 1024 * 1024])
def test_cache_resident_or_unknown_same_launch_count_keeps_native(cache_bytes):
    builder, common, original = case((7, 7), (115, 257), (2, 3), capacity=512)
    builder._hcu_workspace_l2_bytes = cache_bytes
    assert coalesce(builder, common, original) is original


def test_fewer_launches_can_still_coalesce_cache_resident_contexts():
    builder, common, original = case((7, 7, 9, 9), (115, 257, 133, 389), (2, 3, 1, 2), capacity=700)
    builder._hcu_workspace_l2_bytes = 8 * 1024 * 1024
    result = coalesce(builder, common, original)
    assert len(result.prefill.chunks) == 1 < len(original.prefill.chunks)


def test_decode_rows_and_payload_are_unchanged():
    builder, common, original = case((3, 7, 7), (10, 60, 85), (1, 2, 3), capacity=90)
    original.num_decodes = original.num_decode_tokens = 1
    original.num_prefills = 2
    original.num_prefill_tokens = 5
    original.decode = object()
    result = coalesce(builder, common, original)
    assert result.decode is original.decode
    assert result.num_decode_tokens == 1
    assert result.prefill.request_ids.tolist() == [-1, 0, 0, 0, 0, 0]
    assert result.prefill.chunks[0].tokens_slice == slice(1, 6)


def test_decode_only_and_empty_prefill_are_unchanged():
    builder, common, original = case()
    original.num_prefill_tokens = 0
    assert coalesce(builder, common, original) is original
    original.prefill = None
    assert coalesce(builder, common, original) is original


def test_collective_padding_decode_row_does_not_shift_prefill_ids():
    builder, common, original = case((2, 3, 7, 7), (1, 10, 60, 85), (0, 1, 2, 3), capacity=90)
    original.num_decodes = original.num_decode_tokens = 1
    original.num_prefills = 2
    original.num_prefill_tokens = 5
    result = coalesce(builder, common, original, num_decode_rows=2)
    assert result.prefill.request_ids.tolist() == [-1, 0, 0, 0, 0, 0]
    assert result.prefill.chunks[0].chunk_tot_seqlen == 85


@pytest.mark.parametrize('disabled', [False, True])
def test_attention_factory_default_and_kill_leave_native_owner(monkeypatch, disabled):
    from hcu_vllm_das import hy4_prefill_workspace as module
    from vllm.models.hy_v4.nvidia import attention
    calls = []
    marker = object()
    monkeypatch.setattr(attention, 'MLAAttention', lambda *a, **kw: calls.append((a, kw)) or marker)
    if disabled:
        monkeypatch.setenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '1')
        monkeypatch.setenv('HCU_DISABLE_OPS', 'hy4_prefill_workspace')
    else:
        monkeypatch.delenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', raising=False)
    assert module.make_attention(17, named=23) is marker
    assert calls == [((17,), {'named': 23})]


def test_attention_factory_rejects_invalid_policy_before_construction(monkeypatch):
    from hcu_vllm_das import hy4_prefill_workspace as module
    from vllm.models.hy_v4.nvidia import attention
    monkeypatch.setenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', 'invalid')
    monkeypatch.setattr(attention, 'MLAAttention', lambda *a, **kw: pytest.fail('constructed'))
    with pytest.raises(ValueError, match='0 or 1'):
        module.make_attention()


@pytest.mark.parametrize('mode', ['enabled', 'no_pcp', 'dcp', 'foreign', 'bf16'])
def test_attention_factory_composes_only_the_supported_local_owner(monkeypatch, mode):
    import sys
    from hcu_vllm_das import hy4_prefill_workspace as module
    from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as adapter
    from vllm import config
    from vllm.models.hy_v4.nvidia import attention
    base, owned, foreign = type('Base', (), {}), type('Owned', (), {}), type('Foreign', (), {})
    layer = NS(attn_backend=foreign if mode == 'foreign' else base,
               impl=NS(kv_cache_dtype='bf16' if mode == 'bf16' else 'fp8_ds_mla'))
    cfg = NS(parallel_config=NS(prefill_context_parallel_size=1 if mode == 'no_pcp' else 16,
                               decode_context_parallel_size=2 if mode == 'dcp' else 1))
    monkeypatch.setenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '1')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(attention, 'MLAAttention', lambda *a, **kw: layer)
    monkeypatch.setattr(config, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(adapter, 'HcuSparseFlashMLABackend', base, raising=False)
    monkeypatch.setitem(sys.modules, 'hcu_vllm_das.hy4_prefill_backend', NS(HcuHY4PCPWorkspaceBackend=owned))
    before = layer.attn_backend
    result = module.make_attention()
    assert result is layer
    assert result.attn_backend is (owned if mode == 'enabled' else before)
    assert adapter.HcuSparseFlashMLABackend is base
