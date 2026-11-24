# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native Indexer metadata replay with CPU device-kernel leaves isolated."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def native_metadata_on_cpu(monkeypatch):
    from vllm.v1.attention.backends.mla import indexer
    from vllm.utils import torch_utils
    monkeypatch.setattr(torch_utils, 'PIN_MEMORY', False)

    def cpu_kernel(query_cu, seqs, kv_cu, row_cu, mapping, ks, ke,
                   start, stop, rank, world, interleave, *, num_reqs, COMPRESS_RATIO):
        assert rank == 0 and world == 1 and COMPRESS_RATIO == 1
        for r in range(num_reqs):
            mapping[int(kv_cu[r]):int(kv_cu[r+1])] = r
            qbegin, qend = int(query_cu[r]), int(query_cu[r+1])
            for q in range(max(qbegin, start), min(qend, stop)):
                ks[q-start] = row_cu[r]
                ke[q-start] = row_cu[r] + seqs[r] - (qend-qbegin) + q-qbegin + 1

    monkeypatch.setattr(indexer, '_BUILD_PREFILL_CHUNK_METADATA_KERNEL', cpu_kernel)


def case(ids=(7, 7), lengths=(35, 51), queries=(3, 3), *, slices=None,
         request_slice=None, prefix_rows=0):
    from vllm.v1.attention.backends.mla.indexer import (
        build_prefill_chunk_metadata, DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )
    n = len(ids)
    qcu = torch.tensor([0, *np.cumsum(queries).tolist()], dtype=torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32)
    # Real requests have distinct pages; each request's virtual rows alias it.
    width = max(lengths)//4 + 1
    storage = torch.tensor([[i*100000 + j for j in range(width)]
                            for i in range(prefix_rows)] +
                           [[i*100000 + j for j in range(width)] for i in ids], dtype=torch.int32)
    tables = storage[prefix_rows:]
    common = NS(req_idx=np.array(ids), query_start_loc=qcu, query_start_loc_cpu=qcu,
                seq_lens=lens, seq_lens_cpu_upper_bound=lens,
                block_table_tensor=tables, num_reqs=n, num_actual_tokens=int(qcu[-1]),
                mm_req_doc_ranges=None)
    lo, hi = (0, n) if request_slice is None else request_slice
    slices = [None] if slices is None else slices
    chunks = [build_prefill_chunk_metadata(
        lo, hi, qcu, qcu, lens, lens, lens, tables, 1, query_slice=s,
    ) for s in slices]
    extra = DeepseekV32IndexerMetadata(lens, max(lengths), torch.arange(int(qcu[-1])),
        num_decodes=0, num_decode_tokens=0, num_prefills=n,
        num_prefill_tokens=int(qcu[-1]),
        prefill=DeepseekV32IndexerPrefillMetadata(chunks, max(lengths), None))
    builder = NS(use_pcp=True, dcp_world_size=1, compress_ratio=1,
                 indexer_uses_fp4=False, device=torch.device('cpu'),
                 max_prefill_buffer_size=1000,
                 vllm_config=NS(scheduler_config=NS(async_scheduling=False),
                                 attention_config=NS(hisparse_config=None)))
    return builder, common, extra


def apply(builder, common, metadata):
    from hcu_vllm_das.hy4_indexer_workspace import coalesce_indexer_prefill
    return coalesce_indexer_prefill(builder, common, metadata)


def visible_pages(chunk, token):
    row = int(torch.searchsorted(chunk.cu_seq_lens[1:], chunk.cu_seqlen_ks[token], right=True))
    width = int(chunk.cu_seqlen_ke[token]-chunk.cu_seqlen_ks[token])
    logical = torch.arange(width)
    return torch.stack((chunk.block_table[row, logical//4], logical % 4), dim=-1)


def assert_equivalent(original, result):
    assert result.num_prefills == original.num_prefills
    assert result.num_prefill_tokens == original.num_prefill_tokens
    assert result.slot_mapping is original.slot_mapping
    assert sum(c.token_end-c.token_start for c in original.prefill.chunks) == sum(
        c.token_end-c.token_start for c in result.prefill.chunks)
    for new in result.prefill.chunks:
        for token in range(new.token_end-new.token_start):
            absolute = new.token_start + token
            old = next(c for c in original.prefill.chunks if c.token_start <= absolute < c.token_end)
            assert torch.equal(visible_pages(old, absolute-old.token_start), visible_pages(new, token))
        assert new.local_cu_seq_lens is new.cu_seq_lens
        assert new.block_table.is_contiguous()
        assert new.token_to_seq.tolist() == [r for r, (lo, hi) in enumerate(zip(
            new.cu_seq_lens[:-1].tolist(), new.cu_seq_lens[1:].tolist())) for _ in range(hi-lo)]
        assert len(new.token_to_seq) == new.total_seq_lens == new.local_total_seq_lens


def test_native_metadata_two_segments_share_one_prefix():
    b, c, old = case()
    new = apply(b, c, old)
    assert new is not old
    assert old.prefill.chunks[0].total_seq_lens == 86
    chunk, = new.prefill.chunks
    assert chunk.total_seq_lens == 51 and chunk.num_reqs == 1
    assert chunk.cu_seq_lens.tolist() == [0, 51]
    assert chunk.cu_seqlen_ke.tolist() == [33, 34, 35, 49, 50, 51]
    assert old.prefill.chunks[0].cu_seqlen_ke.tolist() == [33, 34, 35, 84, 85, 86]
    assert_equivalent(old, new)


def test_unequal_requests_and_nonzero_request_view_offset():
    b, c, old = case(ids=(3, 7, 7, 9, 9), lengths=(21, 35, 51, 47, 42),
                     queries=(1, 3, 2, 2, 3), prefix_rows=2, request_slice=(1, 5))
    new = apply(b, c, old)
    assert new.prefill.chunks[0].cu_seq_lens.tolist() == [0, 51, 98]
    assert torch.equal(new.prefill.chunks[0].block_table, c.block_table_tensor[[2, 3]])
    assert_equivalent(old, new)


def test_continuations_reuse_whole_chain_layout_with_invisible_groups():
    b, c, old = case(ids=(7, 7, 9, 9), lengths=(35, 51, 47, 42),
                     queries=(3, 3, 2, 3), slices=[slice(0, 2), slice(2, 7), slice(7, 11)])
    new = apply(b, c, old)
    chunks = new.prefill.chunks
    assert chunks[1].skip_kv_gather and chunks[2].skip_kv_gather
    assert [x.num_reqs for x in chunks] == [2, 2, 2]
    assert all(x.block_table is chunks[0].block_table for x in chunks)
    assert all(x.cu_seq_lens is chunks[0].cu_seq_lens for x in chunks)
    assert all(x.token_to_seq is chunks[0].token_to_seq for x in chunks)
    assert_equivalent(old, new)


def test_zero_query_segment_can_supply_longest_context():
    b, c, old = case(lengths=(51, 35), queries=(0, 3))
    new = apply(b, c, old)
    assert new.prefill.chunks[0].total_seq_lens == 51
    assert new.prefill.chunks[0].cu_seqlen_ke.tolist() == [33, 34, 35]
    assert_equivalent(old, new)


@pytest.mark.parametrize('segment_rows,expected_changed', [(80, True), (189, True), (200, False)])
def test_actual_native_128mib_chunker_controls_eligibility(segment_rows, expected_changed):
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder, build_prefill_chunk_metadata,
    )
    b, c, old = case(lengths=(79354, 84835), queries=(segment_rows, segment_rows))
    b.max_prefill_buffer_size = 440000
    specs = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        c.seq_lens_cpu_upper_bound, torch.diff(c.query_start_loc_cpu), 440000, 128*1024**2)
    old.prefill.chunks = [build_prefill_chunk_metadata(
        req.start, req.stop, c.query_start_loc, c.query_start_loc_cpu,
        c.seq_lens, c.seq_lens, c.seq_lens_cpu_upper_bound, c.block_table_tensor,
        1, query_slice=query) for req, query in specs]
    new = apply(b, c, old)
    assert (new is not old) == expected_changed
    assert_equivalent(old, new)
    if segment_rows == 189:
        assert [x.num_reqs for x in old.prefill.chunks] == [1, 1]
        assert len(new.prefill.chunks) == 1
        assert new.prefill.chunks[0].token_end == 378
        assert new.prefill.chunks[0].total_seq_lens == 84835


def test_adjacent_native_chunks_merge_only_within_one_real_request():
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder, build_prefill_chunk_metadata,
    )
    b, c, old = case((7, 7, 9, 9), (35, 51, 37, 53), (3, 3, 3, 3), prefix_rows=2)
    b.max_prefill_buffer_size = 60
    specs = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        c.seq_lens_cpu_upper_bound, torch.diff(c.query_start_loc_cpu), 60, 128*1024**2)
    old.prefill.chunks = [build_prefill_chunk_metadata(
        req.start, req.stop, c.query_start_loc, c.query_start_loc_cpu,
        c.seq_lens, c.seq_lens, c.seq_lens_cpu_upper_bound, c.block_table_tensor,
        1, query_slice=query) for req, query in specs]
    assert len(old.prefill.chunks) == 4
    new = apply(b, c, old)
    assert len(new.prefill.chunks) == 2
    assert [(x.token_start, x.token_end, x.total_seq_lens) for x in new.prefill.chunks] == [(0, 6, 51), (6, 12, 53)]
    assert_equivalent(old, new)


@pytest.mark.parametrize('partial_fit', [False, True])
def test_three_adjacent_segments_respect_greedy_budget(partial_fit, monkeypatch):
    from vllm import envs
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder, build_prefill_chunk_metadata,
    )
    budget = 1024**2
    monkeypatch.setattr(envs, 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', 1)
    lengths = (8000, 8000, 8000) if partial_fit else (20, 30, 40)
    queries = (17, 17, 10) if partial_fit else (3, 3, 3)
    capacity = 8000 if partial_fit else 40
    b, c, old = case((7, 7, 7), lengths, queries)
    b.max_prefill_buffer_size = capacity
    specs = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        c.seq_lens_cpu_upper_bound, torch.diff(c.query_start_loc_cpu), capacity, budget)
    old.prefill.chunks = [build_prefill_chunk_metadata(
        req.start, req.stop, c.query_start_loc, c.query_start_loc_cpu,
        c.seq_lens, c.seq_lens, c.seq_lens_cpu_upper_bound, c.block_table_tensor,
        1, query_slice=query) for req, query in specs]
    assert len(old.prefill.chunks) == 3
    new = apply(b, c, old)
    assert [x.token_end-x.token_start for x in new.prefill.chunks] == ([17, 27] if partial_fit else [9])
    assert all((x.token_end-x.token_start)*x.total_seq_lens*4 <= budget for x in new.prefill.chunks)
    assert_equivalent(old, new)


@pytest.mark.parametrize('mode', ['no_pcp', 'dcp', 'compressed', 'fp4', 'decode',
    'async', 'missing_ids', 'unique_ids', 'nonadjacent', 'negative_id',
    'copied_table', 'strided_table', 'bad_continuation', 'missing_first',
    'incomplete_chain', 'capacity', 'zero_length', 'multimodal', 'hisparse'])
def test_unsupported_or_unproven_layout_keeps_exact_native_metadata(mode):
    b, c, old = case(slices=[slice(0, 3), slice(3, 6)])
    if mode == 'no_pcp': b.use_pcp = False
    if mode == 'dcp': b.dcp_world_size = 2
    if mode == 'compressed': b.compress_ratio = 2
    if mode == 'fp4': b.indexer_uses_fp4 = True
    if mode == 'decode': old.num_decodes = 1
    if mode == 'async': b.vllm_config.scheduler_config.async_scheduling = True
    if mode == 'missing_ids': c.req_idx = None
    if mode == 'unique_ids': c.req_idx = np.array([7, 9])
    if mode == 'nonadjacent': b, c, old = case((7, 9, 7), (35, 42, 51), (2, 3, 1))
    if mode == 'negative_id': c.req_idx = np.array([-1, -1])
    if mode == 'copied_table': old.prefill.chunks[1].block_table = old.prefill.chunks[1].block_table.clone()
    if mode == 'strided_table': old.prefill.chunks[1].block_table = old.prefill.chunks[1].block_table[:, ::2]
    if mode == 'bad_continuation': old.prefill.chunks[1].token_start += 1
    if mode == 'missing_first': old.prefill.chunks.pop(0)
    if mode == 'incomplete_chain': old.prefill.chunks.pop()
    if mode == 'capacity': b.max_prefill_buffer_size = 50
    if mode == 'zero_length': c.seq_lens_cpu_upper_bound = torch.tensor([0, 51])
    if mode == 'multimodal': c.mm_req_doc_ranges = {7: [(0, 3)]}
    if mode == 'hisparse': b.vllm_config.attention_config.hisparse_config = object()
    assert apply(b, c, old) is old


def test_owned_backend_retains_native_abi_and_model_forward(monkeypatch):
    from hcu_vllm_das import hy4_indexer_backend as owned
    from hcu_vllm_das.models import hy4_static
    native = owned.DeepseekV32IndexerBackend
    assert owned.HcuHY4WorkspaceIndexerBackend.get_name() == native.get_name()
    assert (owned.HcuHY4WorkspaceIndexerBackend.get_supported_kernel_block_sizes
            is native.get_supported_kernel_block_sizes)
    assert owned.HcuHY4WorkspaceIndexer.forward is owned.Indexer.forward
    assert owned.HcuHY4WorkspaceIndexer.__init__.__code__ is owned.Indexer.__init__.__code__
    assert (owned.Indexer.__init__.__globals__['DeepseekV32IndexerCache']
            is owned.DeepseekV32IndexerCache)
    assert (owned.HcuHY4WorkspaceIndexer.__init__.__globals__['DeepseekV32IndexerCache']
            is owned.HcuHY4WorkspaceIndexerCache)
    assert (hy4_static.HcuHY4StaticAttention.__init__.__globals__['Indexer']
            is hy4_static.make_indexer)
    assert hy4_static.attention.HYV4MLAAttention.__init__.__globals__['Indexer'] is owned.Indexer
    cache = owned.HcuHY4WorkspaceIndexerCache.__new__(owned.HcuHY4WorkspaceIndexerCache)
    backend = cache.get_attn_backend()
    assert backend.get_builder_cls() is owned.HcuHY4WorkspaceIndexerBuilder
    b, c, original = case()
    calls = []
    monkeypatch.setattr(owned.DeepseekV32IndexerMetadataBuilder, 'build',
                        lambda self, prefix, common, fast: calls.append((prefix, common, fast)) or original)
    builder = backend.get_builder_cls().__new__(backend.get_builder_cls())
    builder.__dict__.update(vars(b))
    result = builder.build(0, c, fast_build=True)
    assert calls == [(0, c, True)]
    assert_equivalent(original, result)
    assert result is not original


@pytest.mark.parametrize('mode', ['enabled', 'off', 'master_off', 'kill', 'no_pcp',
                                 'dcp', 'async', 'bf16', 'hisparse'])
def test_factory_startup_policy_keeps_native_when_disabled(mode, monkeypatch):
    from hcu_vllm_das import hy4_indexer_backend as owned
    from hcu_vllm_das.hy4_indexer_workspace import make_indexer
    from vllm import config
    from vllm.models.hy_v4.nvidia import attention
    cfg = NS(parallel_config=NS(prefill_context_parallel_size=16, decode_context_parallel_size=1),
             scheduler_config=NS(async_scheduling=False), cache_config=NS(cache_dtype='fp8_e4m3'),
             attention_config=NS(hisparse_config=None))
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '1')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    if mode == 'off': monkeypatch.setenv('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '0')
    if mode == 'master_off': monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    if mode == 'kill': monkeypatch.setenv('HCU_DISABLE_OPS', 'hy4_prefill_workspace')
    if mode == 'no_pcp': cfg.parallel_config.prefill_context_parallel_size = 1
    if mode == 'dcp': cfg.parallel_config.decode_context_parallel_size = 2
    if mode == 'async': cfg.scheduler_config.async_scheduling = True
    if mode == 'bf16': cfg.cache_config.cache_dtype = 'bfloat16'
    if mode == 'hisparse': cfg.attention_config.hisparse_config = object()
    monkeypatch.setattr(config, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(attention, 'Indexer', lambda *args, **kwargs: ('native', args, kwargs))
    monkeypatch.setattr(owned, 'HcuHY4WorkspaceIndexer', lambda *args, **kwargs: ('owned', args, kwargs))
    result = make_indexer(3, prefix='model.layers.0.indexer')
    assert result == ('owned' if mode == 'enabled' else 'native',
                      (3,), {'prefix': 'model.layers.0.indexer'})
