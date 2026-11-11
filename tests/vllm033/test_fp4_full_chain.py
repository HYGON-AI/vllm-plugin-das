# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real 033 constructors and reader dispatch with CPU/native leaf doubles.

No Triton JIT is compiled or executed. Providers and GPU leaf kernels are
recorders. The real selector, constructors, cache spec, reader orchestration,
local top-k and tensor address/stride operations are under test.
"""
from types import SimpleNamespace as NS
from dataclasses import fields
import importlib

import pytest
import torch
from hcu_vllm_das import fp4_indexer as impl
from hcu_vllm_das import fp4_indexer_lightop as reader
from hcu_vllm_das import v41_indexer as scorer
from hcu_vllm_das import v41_indexer_activation as activation
from hcu_vllm_das import fp4_packer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from test_fp4_full_admission import attention_config, config


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.output_size = sum(output_size) if isinstance(output_size, list) else output_size
        self.calls = []
    def forward(self, x):
        self.calls.append(x)
        out = torch.zeros((x.shape[0], self.output_size), dtype=torch.bfloat16)
        out[:, 0] = 1
        return out, None


class Norm(torch.nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = torch.ones(width, dtype=torch.bfloat16)
        self.variance_epsilon = eps


class CaptureKernel:
    def __init__(self):
        self.calls = []
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


@pytest.fixture
def chain(monkeypatch):
    cfg = config()
    hf = cfg.model_config.hf_config
    hf.__dict__.update(hidden_size=128, q_lora_rank=128, o_lora_rank=16,
                       num_attention_heads=1, head_dim=128, o_groups=1,
                       sliding_window=16, kv_source_layer_ids=[0], index_source_layer_ids=[0],
                       candidate_source_layer_id=-1, candidate_topk_blocks=0,
                       candidate_block_size=0, num_hidden_layers=2, rms_norm_eps=1e-6,
                       max_position_embeddings=4096, index_topk=4)
    hf.compress_ratios = [1, 0]
    cfg.compilation_config = NS(static_forward_context={})
    cfg.quant_config = None
    cfg.cache_config.cache_dtype = 'fp8_ds_mla'
    cfg.cache_config.swa_bounded_replay = False
    cfg.cache_config.enable_prefix_caching = False
    cfg.scheduler_config = NS(max_num_batched_tokens=16, max_num_seqs=4)
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'lightop')
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'triton')
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'stock')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    # This chain exercises the FP4 path, so the deployment has to declare the
    # MXFP4 indexer cache: `fp4_indexer.applicable()` now asks core whether the
    # model actually uses one, and core resolves `auto` to `fp8`.
    cfg.attention_config = attention_config('mxfp4')
    monkeypatch.setattr(activation, 'check_health', lambda: None)
    monkeypatch.setattr(fp4_packer, 'check_health', lambda: None)
    monkeypatch.setattr(fp4_packer, '_state', dict(mode='triton'))
    monkeypatch.setattr(impl, 'load_providers', lambda: reader.providers())
    monkeypatch.setattr(impl, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(scorer, 'current_config', lambda: cfg)
    monkeypatch.setattr(scorer, 'register_splitting_op', lambda cfg: None)
    sparse_layer = importlib.import_module('vllm.model_executor.layers.sparse_attn_indexer')
    monkeypatch.setattr(sparse_layer, 'get_current_vllm_config', lambda: cfg)
    cfg.kernel_config.sparse_indexer_topk_backend = 'torch'
    cfg.parallel_config.pcp_shard_decode_requests = False
    monkeypatch.setattr(impl.common, '_use_v41_mxfp8_kv_record', lambda: False)
    context = NS(attn_metadata={})
    monkeypatch.setattr(impl, 'get_forward_context', lambda: context)
    monkeypatch.setattr(reader, 'get_forward_context', lambda: context)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'Event', lambda: object())
    from vllm.model_executor import custom_op
    monkeypatch.setattr(custom_op, 'op_registry_oot', {'SparseAttnIndexer': scorer.HcuV41SparseAttnIndexer})
    # CustomOp's device selector is outside this CPU test. Call its real HIP
    # implementation after real constructor admission, never a CUDA dispatch.
    monkeypatch.setattr(custom_op.CustomOp, '__init__', torch.nn.Module.__init__)
    constructor_spaces = (impl._COMMON_INIT.__globals__, impl._INDEXER_INIT.__globals__,
                          impl.Fp4IndexerCache.__init__.__globals__)
    for namespace in constructor_spaces:
        for name in ('get_current_vllm_config',):
            if name in namespace:
                monkeypatch.setitem(namespace, name, lambda: cfg)
        for name in ('ReplicatedLinear', 'MergedColumnParallelLinear',
                     'ColumnParallelLinear', 'RowParallelLinear'):
            if name in namespace:
                monkeypatch.setitem(namespace, name, Linear)
        if 'RMSNorm' in namespace:
            monkeypatch.setitem(namespace, 'RMSNorm', Norm)
        if 'get_max_prefill_buffer_size' in namespace:
            monkeypatch.setitem(namespace, 'get_max_prefill_buffer_size', lambda cfg: 128)
    ns = impl._COMMON_INIT.__globals__
    monkeypatch.setitem(ns, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setitem(ns, 'make_attention_sink', lambda *args: torch.zeros(1))
    rope = NS(cos_sin_cache=torch.cat((torch.ones(4096, 32), torch.zeros(4096, 32)), -1))
    monkeypatch.setitem(ns, 'build_deepseek_v4_rope', lambda *args, **kwargs: rope)
    monkeypatch.setitem(ns, '_resolve_dsv4_kv_cache_dtype', lambda *args: ('fp8_ds_mla', torch.uint8))
    monkeypatch.setitem(ns, '_use_v41_mxfp8_kv_record', lambda: False)
    class Empty(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    monkeypatch.setitem(ns, 'DeepseekV4SWACache', Empty)
    monkeypatch.setitem(ns, 'DeepseekCompressor', Empty)
    monkeypatch.setattr(impl.rocm, '_aiter_sparse_mla_enabled', lambda *args: False)
    calls = []
    def gather(cache, slots, page_size):
        calls.append(('gather', cache, slots.clone(), page_size))
        return torch.zeros((slots.numel(), 64), dtype=torch.int8), torch.zeros(slots.numel(), dtype=torch.int32)
    def prefill(q, kv, weights, starts, ends, clean, width, candidates, lens):
        calls.append(('prefill', q, kv, weights, starts, ends, clean, width))
        out = torch.full((q[0].shape[0], width + 8), -float('inf'))[:, :width]
        for row, n in enumerate((ends-starts).tolist()):
            out[row, :n] = torch.arange(n).float()
        return out
    def decode(q, sf, cache, weights, lens, table, schedule, width, block, out, rows):
        calls.append(('decode', q, sf, cache, weights, lens.clone(), table, schedule, block, rows))
        logits = torch.full((q.shape[0], width + 8), -float('inf'))[:, :width]
        for row, n in enumerate(lens.tolist()):
            logits[row, :n] = torch.arange(n).float()
        return logits
    monkeypatch.setattr(reader, '_providers', reader.Providers(gather, prefill, decode, object()))
    kernel = CaptureKernel()
    monkeypatch.setattr(impl.store, '_indexer_k_norm_rope_quant_store_kernel', kernel)
    return NS(cfg=cfg, context=context, calls=calls, kernel=kernel, rope=rope)


def make_attention(chain):
    # The unmodified native AMD __init__ runs, including its native super call.
    # Its super now reaches the private common constructor through the MRO.
    return impl.Fp4Attention(vllm_config=chain.cfg, prefix='model.layers.0.self_attn',
                             topk_indices_buffer=torch.full((16, 4), -77, dtype=torch.int32))


def metadata(chain, cache, decode_lens=(3, 5), prefill=False):
    n = len(decode_lens)
    kw = dict(seq_lens=torch.tensor(decode_lens, dtype=torch.int32),
              max_seq_len=4096, slot_mapping=torch.tensor([0, -1], dtype=torch.int64),
              num_decodes=n, num_decode_tokens=n, num_prefills=0, num_prefill_tokens=0,
              prefill=None, decode=NS(block_table=torch.tensor([[1], [0]], dtype=torch.int32)[:n],
                  seq_lens=torch.tensor(decode_lens, dtype=torch.int32).unsqueeze(1),
                  decode_lens=torch.ones(n, dtype=torch.int32), requires_padding=False),
              block_table=torch.tensor([[1], [0]], dtype=torch.int32)[:n])
    if prefill:
        kw.update(num_decodes=0, num_decode_tokens=0, num_prefills=2,
                  num_prefill_tokens=2, decode=None)
        chunk = NS(token_start=0, token_end=2, total_seq_lens=5,
                   token_to_seq=torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
                   cu_seq_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
                   block_table=torch.tensor([[1], [0]], dtype=torch.int32),
                   cu_seqlen_ks=torch.tensor([0, 2], dtype=torch.int32),
                   cu_seqlen_ke=torch.tensor([2, 5], dtype=torch.int32))
        kw['prefill'] = NS(chunks=[chunk])
    obj = DeepseekV32IndexerMetadata(**kw)
    chain.context.attn_metadata[cache.prefix] = obj
    return obj


def query(rows=2):
    return reader.Fp4Query(torch.zeros((rows, 32, 64), dtype=torch.uint8),
                           torch.full((rows, 32), 0x7f7f7f7f, dtype=torch.int32),
                           torch.ones((rows, 32, 128), dtype=torch.bfloat16))


def bind_cache(cache):
    backing = torch.full((2, 4608), 0xA5, dtype=torch.uint8)
    view = backing.as_strided((2, 1, 64, 68), (4608, 4352, 68, 1))
    cache.bind_kv_cache(view)
    return backing


def test_fp4_backend_uses_logical_64_for_ratio2_physical_32_page():
    ratio2 = NS(tokens_per_state=2)
    assert impl.Fp4IndexerBackend.get_supported_kernel_block_sizes(ratio2) == [64]
    assert impl.Fp4IndexerBackend.get_supported_kernel_block_sizes() == [64]
    backing = torch.zeros((2, 32 * 68), dtype=torch.uint8)
    cache = reader.cache_view(backing.as_strided((2, 32, 68), (32 * 68, 68, 1)))
    assert reader.physical_page_records(cache) == 32


def test_real_attention_constructor_selects_68byte_cache_before_binding(chain):
    attention = make_attention(chain)
    assert isinstance(attention.indexer, impl.Fp4Indexer)
    cache = attention.indexer.k_cache
    assert isinstance(cache, impl.Fp4IndexerCache)
    assert cache.head_dim == 68 and not cache.rocm_mxfp4
    spec = cache.get_kv_cache_spec(chain.cfg)
    assert spec.head_size == 68 and spec.num_states == 64
    assert spec.page_size_bytes >= 4352
    assert cache.get_attn_backend().get_builder_cls() is impl.Fp4MetadataBuilder
    assert isinstance(attention.indexer.indexer_op, scorer.HcuV41SparseAttnIndexer)
    assert attention.indexer.indexer_op._hcu_fp4
    assert not torch.cuda.is_initialized()


def test_mixed_ratio_model_is_admitted_before_allocation(chain):
    chain.cfg.model_config.hf_text_config.compress_ratios = [0, 1, 2]
    plan = impl.validate_config(chain.cfg)
    assert plan.ratios == (1, 2)


def test_k_store_reuses_official_kernel_with_real_pitch_and_slots(chain):
    indexer = make_attention(chain).indexer
    backing = bind_cache(indexer.k_cache)
    meta = metadata(chain, indexer.k_cache)
    latent = torch.ones((2, 128), dtype=torch.bfloat16)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    indexer._produce_k(latent, positions, chain.rope)
    grid, args, kw = chain.kernel.calls[0]
    assert grid == (2,)
    assert args[2] is positions and args[8] is meta.slot_mapping
    assert kw['KV_BLOCK_STRIDE'] == 4608 and args[9] == 64
    assert kw['USE_FP4'] is True and kw['SHUFFLE'] is False
    assert kw['SCALE_DIM'] == 4 and kw['TOKEN_STRIDE'] == 64
    assert kw['MX_SCALE'] is False
    assert backing[:, 4352:].eq(0xA5).all()


def test_q_pack_actual_core_wrapper_reaches_native_leaf_and_retains_bf16(chain, monkeypatch):
    indexer = make_attention(chain).indexer
    qmod = importlib.import_module('vllm.models.deepseek_v4.common.ops.fused_indexer_q')
    calls = []
    def leaf(positions, q, rope, weights, softmax, head, packed, sf, out):
        calls.append((positions, q, rope, softmax, head))
        packed.fill_(0x21)
        sf.fill_(127)
        out.copy_(weights.float() * softmax * head)
    monkeypatch.setattr(qmod, 'has_cutedsl', lambda: False)
    monkeypatch.setattr(qmod, 'current_platform', NS(is_xpu=lambda: False))
    monkeypatch.setattr(qmod, '_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL', leaf)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.ones((2, 32), dtype=torch.bfloat16)
    result, scale, folded = indexer.forward_q(torch.ones((2, 128), dtype=torch.bfloat16),
                                             None, weights, positions, chain.rope)
    assert scale is None and isinstance(result, reader.Fp4Query)
    assert result[0].eq(0x21).all()
    assert result[1].dtype == torch.int32 and result[1].eq(0x7f7f7f7f).all()
    assert result.bf16.dtype == torch.bfloat16 and result.bf16.shape == (2, 32, 128)
    assert torch.equal(folded, weights.float() * 128**-0.5 * 32**-0.5)
    assert calls[0][0] is positions


@pytest.mark.parametrize('prefill', [False, True])
def test_actual_oot_forward_reaches_both_readers_with_padded_strides(chain, prefill):
    indexer = make_attention(chain).indexer
    backing = bind_cache(indexer.k_cache)
    metadata(chain, indexer.k_cache, prefill=prefill)
    out = indexer.indexer_op.forward_hip(torch.empty((2, 128)), query(), None,
                                        torch.ones((2, 32), dtype=torch.float32))
    if prefill:
        gather, call = chain.calls
        assert gather[0] == 'gather' and gather[1].stride(0) == 4608
        assert gather[2].tolist() == [64, 65, 0, 1, 2]
        assert call[0] == 'prefill' and call[1][0].dtype == torch.bfloat16
        assert call[1][1] is None and call[6] is False
        assert out[:2].tolist() == [[0, 1, -1, -1], [0, 1, 2, -1]]
    else:
        call = chain.calls[0]
        assert call[0] == 'decode' and call[1].shape == (2, 1, 32, 64)
        assert call[2].shape == (2, 1, 32) and call[2].dtype == torch.int32
        assert call[3].stride(0) == 4608
        assert call[3].untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        assert call[7] is None and call[8:] == (64, 1)
        assert out[0].tolist() == [0, 1, 2, -1]
        assert set(out[1].tolist()) == {1, 2, 3, 4}
    assert backing[:, 4352:].eq(0xA5).all()


@pytest.mark.parametrize('page_records', [32, 64, 128])
def test_decode_staging_preserves_payload_scale_and_fast_path(page_records):
    rows, max_context = 2, 128
    allocator_pages_per_row = (max_context + page_records - 1) // page_records
    staged_pages_per_row = (max_context + 63) // 64
    pages = rows * allocator_pages_per_row
    backing = torch.arange(pages * page_records * 68, dtype=torch.int64).remainder(256).to(
        torch.uint8
    ).reshape(pages, page_records * 68)
    cache = backing.as_strided(
        (pages, page_records, 68), (page_records * 68, 68, 1)
    )
    table = torch.arange(pages, dtype=torch.int32).reshape(rows, allocator_pages_per_row)
    lengths = torch.tensor([70, 65], dtype=torch.int32)
    calls = []

    def gather(flat, slots, page_size):
        calls.append((flat, slots.clone(), page_size))
        physical = flat.reshape(pages, page_size * 68)
        page_id = torch.div(slots, page_size, rounding_mode='floor')
        row = torch.remainder(slots, page_size)
        payload_offset = page_id * page_size * 68 + row * 64
        scale_offset = page_id * page_size * 68 + page_size * 64 + row * 4
        payload_bytes = torch.stack(
            [physical.reshape(-1)[payload_offset + col] for col in range(64)], dim=1
        )
        scale_bytes = torch.stack(
            [physical.reshape(-1)[scale_offset + col] for col in range(4)], dim=1
        )
        return payload_bytes.contiguous().view(torch.int8), scale_bytes.contiguous().view(torch.int32).reshape(-1)

    cache = reader.cache_view(cache)
    staged, staged_table = reader.stage_decode_pages(
        cache, table, lengths, max_context, gather
    )
    if page_records == 64:
        assert staged is cache
        assert staged_table is table
        assert calls == []
        return
    assert staged.shape == (rows * staged_pages_per_row, 64, 68)
    assert staged_table.tolist() == torch.arange(
        rows * staged_pages_per_row, dtype=torch.int32
    ).reshape(rows, staged_pages_per_row).tolist()
    assert calls and calls[0][2] == page_records
    assert staged.dtype == torch.uint8
    expected_payload = torch.empty((rows, staged_pages_per_row, 4096), dtype=torch.uint8)
    expected_scales = torch.empty((rows, staged_pages_per_row, 256), dtype=torch.uint8)
    for request in range(rows):
        for page in range(staged_pages_per_row):
            for offset in range(64):
                logical = page * 64 + offset
                if logical < int(lengths[request]):
                    physical_page = int(table[request, logical // page_records])
                    physical_row = logical % page_records
                else:
                    physical_page, physical_row = 0, 0
                rec = backing[physical_page]
                expected_payload[request, page, offset * 64:(offset + 1) * 64] = rec[
                    physical_row * 64:physical_row * 64 + 64
                ]
                expected_scales[request, page, offset * 4:(offset + 1) * 4] = rec[
                    page_records * 64 + physical_row * 4:
                    page_records * 64 + physical_row * 4 + 4
                ]
            staged_flat = staged[request * staged_pages_per_row + page].as_strided((4352,), (1,))
            assert torch.equal(staged_flat[:4096], expected_payload[request, page])
            assert torch.equal(staged_flat[4096:], expected_scales[request, page])


def test_invalid_metadata_fails_before_output_write(chain):
    indexer = make_attention(chain).indexer
    bind_cache(indexer.k_cache)
    meta = metadata(chain, indexer.k_cache)
    meta.decode.requires_padding = True
    before = indexer.topk_indices_buffer.clone()
    with pytest.raises(ValueError, match='flattened'):
        indexer.indexer_op.forward_hip(torch.empty((2, 128)), query(), None, torch.ones((2, 32)))
    assert torch.equal(before, indexer.topk_indices_buffer)
    assert not chain.calls


def test_zero_length_decode_rows_leave_minus_one(chain):
    indexer = make_attention(chain).indexer
    bind_cache(indexer.k_cache)
    metadata(chain, indexer.k_cache, decode_lens=(0, 0))
    out = indexer.indexer_op.forward_hip(torch.empty((2, 128)), query(), None, torch.ones((2, 32)))
    assert out[:2].eq(-1).all()


def test_unsupported_cache_pitch_rejected_without_copy():
    bad = torch.empty((2, 4352), dtype=torch.uint8).as_strided((2, 64, 68), (4351, 68, 1))
    with pytest.raises(ValueError, match='aligned'):
        reader.cache_view(bad)
