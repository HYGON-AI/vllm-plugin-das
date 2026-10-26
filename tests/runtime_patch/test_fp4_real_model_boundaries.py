# SPDX-License-Identifier: Apache-2.0
"""CPU/mock checks of real V4.1 return/metadata and native ABI boundaries."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fp4_indexer import (
    Fp4QTuple,
    fp4_chunk_slots,
    fp4_flat_page_view,
    fp4_physical_page_size,
    stage_fp4_decode_pages,
)
from vllm_hcu.model_executor.layers import fp4_indexer as fp4
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_indexer_k_layout as patch


ROOT = Path(__file__).resolve().parents[2]


def _reader(layer, **replacements):
    path = ROOT / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    names = {"_hcu_fp4_sparse_attn_indexer_native", "_topk_indices_torch",
             "_decode_row_ends_from_seq_lens"}
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch, Fp4QTuple=Fp4QTuple,
              get_forward_context=lambda: SimpleNamespace(attn_metadata={"cache": layer}),
              _resolve_layer_name=lambda x: x, fp4_chunk_slots=fp4_chunk_slots,
              fp4_flat_page_view=fp4_flat_page_view)
    ns.update(fp4_physical_page_size=fp4_physical_page_size,
              stage_fp4_decode_pages=stage_fp4_decode_pages)
    ns.update(replacements)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns["_hcu_fp4_sparse_attn_indexer_native"]


def _query(rows):
    return Fp4QTuple(torch.zeros(rows, 32, 64, dtype=torch.uint8),
                     torch.zeros(rows, 32, dtype=torch.int32),
                     torch.zeros(rows, 32, 128, dtype=torch.bfloat16))


def _run(fn, q, out, **kwargs):
    return fn(None, "cache", torch.zeros(2, 64, 68, dtype=torch.uint8), q,
              None, torch.ones(q.packed.shape[0], 32), 128, "ue8m0", out.shape[1],
              128, 128, 128, out, skip_k_cache_insert=True, **kwargs)


def test_page_view_preserves_offset_padding_and_physical_planes():
    backing = torch.arange(3 * 4800, dtype=torch.int32).to(torch.uint8)
    cache = backing[32:].as_strided((2, 64, 68), (4800, 68, 1))
    flat = fp4_flat_page_view(cache)
    assert flat.data_ptr() == cache.data_ptr()
    assert flat.shape == (2, 4352) and flat.stride() == (4800, 1)
    assert torch.equal(flat[1], backing[4832:9184])


@pytest.mark.parametrize("page_size", [32, 128])
def test_decode_staging_copies_allocator_bytes_without_requantizing(monkeypatch, page_size):
    stride = page_size * 68 + 256
    raw = torch.arange(3 * stride, dtype=torch.int32).to(torch.uint8).reshape(3, stride)
    cache = raw.as_strided((3, page_size, 68), (stride, 68, 1))
    before = raw.clone()
    table = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.int32)
    lengths = torch.tensor([65, 33], dtype=torch.int32)

    def gather(flat, slots, physical_page_size):
        assert physical_page_size == page_size
        pages, pos = slots // page_size, slots % page_size
        packed = flat[pages[:, None], pos[:, None] * 64 + torch.arange(64)]
        sf = flat[pages[:, None], page_size * 64 + pos[:, None] * 4 + torch.arange(4)]
        return packed.contiguous().view(torch.int8), sf.contiguous().view(torch.int32).flatten()

    monkeypatch.setattr(fp4, "lightop_gather_index_k_fp4", gather)
    staged, mapped = stage_fp4_decode_pages(cache, table, lengths, 65)
    assert staged.shape == (4, 64, 1, 68)
    assert mapped.tolist() == [[0, 1], [2, 3]]
    staged_flat = fp4_flat_page_view(staged)
    for row, length in enumerate(lengths.tolist()):
        for col in range(length):
            src_page = int(table[row, col // page_size])
            src_pos = col % page_size
            dst_page = int(mapped[row, col // 64])
            dst_pos = col % 64
            assert torch.equal(staged_flat[dst_page, dst_pos * 64:(dst_pos + 1) * 64],
                               raw[src_page, src_pos * 64:(src_pos + 1) * 64])
            assert torch.equal(staged_flat[dst_page, 4096 + dst_pos * 4:4100 + dst_pos * 4],
                               raw[src_page, page_size * 64 + src_pos * 4:
                                   page_size * 64 + src_pos * 4 + 4])
    assert torch.equal(raw, before)


def test_prefill_second_request_uses_window_local_topk_and_candidates():
    chunk = SimpleNamespace(
        token_start=0, token_end=2, local_total_seq_lens=5,
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        token_to_seq=torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([2, 5], dtype=torch.int32),
    )
    layer = SimpleNamespace(prefill=SimpleNamespace(chunks=[chunk]), decode=None,
                            num_prefills=2, num_decodes=0)
    seen = []

    def policy(logits, starts, ends, candidates, block_size, write):
        assert starts is None
        assert ends.tolist() == [2, 3]
        seen.append((block_size, write))

    def gather(cache, slots, page_size):
        assert slots.tolist() == [0, 1, 64, 65, 66]
        return torch.zeros(5, 64, dtype=torch.int8), torch.zeros(5, dtype=torch.int32)

    fn = _reader(layer, lightop_gather_index_k_fp4=gather,
                 lightop_fp4_prefill=lambda *a, **kw: torch.tensor([[2., 1., -torch.inf],
                                                                  [1., 2., 3.]]),
                 _apply_candidate_policy=policy)
    out = torch.full((2, 1), -9, dtype=torch.int32)
    _run(fn, _query(2), out, candidate_blocks=torch.zeros(2, 4, dtype=torch.int32),
         candidate_block_size=8, candidate_write=False)
    assert out.tolist() == [[0], [2]]
    assert seen == [(8, False)]


def test_flattened_dspark_ratio2_decode_preserves_per_token_bounds_and_prefill_suffix():
    # Two requests, each with six tokens. Expansion precedes ratio2 division.
    expected = torch.tensor([32, 32, 33, 33, 34, 34, 0, 0, 1, 1, 2, 2], dtype=torch.int32)
    decode = SimpleNamespace(seq_lens=expected[:, None], requires_padding=False,
                             block_table=torch.zeros(12, 2, dtype=torch.int32),
                             schedule_metadata=object())  # deliberately no num_decode_tokens
    layer = SimpleNamespace(decode=decode, prefill=None, num_decodes=2,
                            num_prefills=0, num_decode_tokens=12)
    calls = []

    def scorer(q, sf, cache, weights, lens, table, schedule, width, table_size, out, repeat):
        assert q.shape == (12, 1, 32, 64)
        assert sf.shape == (12, 1, 32) and sf.dtype == torch.int32
        assert weights.shape == (12, 32)
        assert torch.equal(lens, expected)
        assert schedule is None and repeat == 1 and table_size == 64
        calls.append(True)
        return torch.arange(width).float().expand(12, -1)

    fn = _reader(layer, lightop_fp4_decode=scorer)
    out = torch.full((15, 1), 99, dtype=torch.int32)
    _run(fn, _query(15), out)
    assert out[:12, 0].tolist() == [31, 31, 32, 32, 33, 33, -1, -1, 0, 0, 1, 1]
    assert out[12:, 0].tolist() == [-1, -1, -1]  # initialized, never used as padded decode rows
    assert calls == [True]


def test_padded_native_decode_rejected_before_scorer():
    decode = SimpleNamespace(requires_padding=True)
    layer = SimpleNamespace(decode=decode, prefill=None, num_decodes=1,
                            num_prefills=0, num_decode_tokens=1)
    fn = _reader(layer)
    with pytest.raises(RuntimeError, match="flattened metadata"):
        _run(fn, _query(1), torch.zeros(1, 1, dtype=torch.int32))


def test_real_forward_boundary_retains_adapter_and_off_delegates(monkeypatch):
    class Indexer:
        def forward(self, qr, latent, indexer_weights, positions, rotary_emb, qr_scale=None):
            return "original"

    q = _query(2)
    weights = torch.ones(2, 32)
    target = SimpleNamespace(
        DeepseekV4Indexer=Indexer,
        get_forward_context=lambda: SimpleNamespace(attn_metadata=None),
        fused_indexer_q_rope_quant=lambda *a, **kw: (q, weights),
    )
    patch._install_fp4_forward(target)
    instance = Indexer()
    instance.use_fp4_kv = True
    instance.owns_k = False
    instance._wq_b_proj = lambda *a: q.bf16
    instance.n_head, instance.head_dim, instance.softmax_scale = 32, 128, 1.0
    instance.indexer_weights_dtype = torch.float32
    rope = SimpleNamespace(cos_sin_cache=torch.empty(0))
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FP4_INDEXER", "1")
    index_q, scale, out_weights = instance.forward(None, None, weights, None, rope)
    # This is the exact upstream attention caller reconstruction.
    transported = (index_q, scale) if scale is not None else index_q
    assert transported is q and transported.bf16 is q.bf16
    assert out_weights is weights
    monkeypatch.setenv("VLLM_HCU_USE_FP4_INDEXER", "0")
    assert instance.forward(None, None, weights, None, rope) == "original"


def _domain_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=SimpleNamespace(compress_ratios=[0, 2, 1],
                                            kv_source_layer_ids=[2, 20],
                                            index_source_layer_ids=[2, 8, 20, 24]),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                         prefill_context_parallel_size=1,
                                         decode_context_parallel_size=1, data_parallel_size=8),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5,
                                           enable_adaptive_verification=False),
    )


def test_real_text_config_dp8_dspark5_domain_and_per_dimension_rejections(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FP4_INDEXER", "1")
    cfg = _domain_config()
    target = SimpleNamespace()
    assert patch._fp4_domain_requested(target, cfg)
    for name, value in (("decode_context_parallel_size", 8), ("prefill_context_parallel_size", 8),
                        ("pipeline_parallel_size", 2), ("tensor_parallel_size", 2),
                        ("data_parallel_size", 4)):
        cfg = _domain_config()
        setattr(cfg.parallel_config, name, value)
        assert not patch._fp4_domain_requested(target, cfg), name
    cfg = _domain_config()
    cfg.speculative_config.num_speculative_tokens = 4
    assert not patch._fp4_domain_requested(target, cfg)
    cfg = _domain_config()
    cfg.kv_transfer_config = object()
    assert not patch._fp4_domain_requested(target, cfg)
    # 真实 PD connector 仍拒绝（无 FP4 indexer cache 传输契约）。
    cfg = _domain_config()
    cfg.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnector")
    assert not patch._fp4_domain_requested(target, cfg)
    cfg = _domain_config()
    cfg.kv_transfer_config = SimpleNamespace(kv_connector="NixlConnector")
    assert not patch._fp4_domain_requested(target, cfg)
    # DecodeBenchConnector 例外：本机 dummy 填充，不跨节点传输。
    cfg = _domain_config()
    cfg.kv_transfer_config = SimpleNamespace(kv_connector="DecodeBenchConnector")
    assert patch._fp4_domain_requested(target, cfg)
    cfg = _domain_config()
    cfg.attention_config = SimpleNamespace(indexer_sparse_logits=True)
    assert not patch._fp4_domain_requested(target, cfg)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert not patch._fp4_domain_requested(target, _domain_config())


@pytest.mark.parametrize("max_model_len,expected_width", [
    (4096, 4096), (4097, 4100), (4098, 4100), (4099, 4100),
])
def test_prefill_aligns_output_width_without_changing_gather_or_row_lengths(
    max_model_len, expected_width
):
    page_count = (max_model_len + 63) // 64
    block_table = torch.zeros(3, page_count, dtype=torch.int32)
    block_table[0] = torch.arange(page_count, dtype=torch.int32)
    block_table[1, 0] = page_count
    starts = torch.tensor([0, max_model_len, max_model_len + 2], dtype=torch.int32)
    ends = torch.tensor([max_model_len, max_model_len + 2, max_model_len + 2],
                        dtype=torch.int32)
    chunk = SimpleNamespace(
        token_start=0, token_end=3, local_total_seq_lens=max_model_len + 2,
        block_table=block_table,
        token_to_seq=torch.cat((torch.zeros(max_model_len, dtype=torch.int32),
                                torch.ones(2, dtype=torch.int32))),
        cu_seq_lens=starts, cu_seqlen_ks=starts, cu_seqlen_ke=ends,
    )
    layer = SimpleNamespace(prefill=SimpleNamespace(chunks=[chunk]), decode=None,
                            num_prefills=3, num_decodes=0)
    calls = []

    def gather(cache, slots, page_size):
        assert page_size == 64
        expected_slots = torch.cat((torch.arange(max_model_len),
                                    torch.tensor([page_count * 64, page_count * 64 + 1])))
        assert torch.equal(slots, expected_slots)
        return (torch.zeros(max_model_len + 2, 64, dtype=torch.int8),
                torch.zeros(max_model_len + 2, dtype=torch.int32))

    def scorer(q, packed, scales, weights, ks, ke, *, max_seqlen_k):
        assert max_seqlen_k == expected_width
        assert max_seqlen_k % 4 == 0
        assert ks is starts and ke is ends
        assert packed.shape[0] == scales.shape[0] == max_model_len + 2
        logits = torch.arange(expected_width, dtype=torch.float32).repeat(3, 1)
        for row, length in enumerate((max_model_len, 2, 0)):
            logits[row, length:] = 1e30
        calls.append(max_seqlen_k)
        return logits

    fn = _reader(layer, lightop_gather_index_k_fp4=gather, lightop_fp4_prefill=scorer)
    cache = torch.zeros(page_count + 1, 64, 68, dtype=torch.uint8)
    out = torch.full((3, 3), -9, dtype=torch.int32)
    fn(None, "cache", cache, _query(3), None, torch.ones(3, 32), 128, "ue8m0", 3,
       128, max_model_len, max_model_len + 2, out, skip_k_cache_insert=True)
    assert calls == [expected_width]
    assert out.tolist() == [[max_model_len - 1, max_model_len - 2, max_model_len - 3],
                            [0, 1, -1], [-1, -1, -1]]
    print(f"P2_RUNTIME max_model_len={max_model_len} scorer_width={calls[0]} "
          f"topk={out[0].tolist()}")


def test_topk_masks_extreme_alignment_padding_and_returns_invalid_sentinel():
    fn = _reader(None)
    topk = fn.__globals__["_topk_indices_torch"]
    logits = torch.arange(4100, dtype=torch.float32).repeat(3, 1)
    row_lens = torch.tensor([4097, 2, 0], dtype=torch.int32)
    for row, length in enumerate(row_lens.tolist()):
        logits[row, length:] = 1e30
    assert bool(torch.all(logits[0, 4097:] > 1e20))
    unmasked = torch.topk(logits[0], k=3).indices
    assert bool(torch.all(unmasked >= 4097))
    selected = topk(logits, 3, row_ends=row_lens)
    assert selected.tolist() == [[4096, 4095, 4094], [0, 1, -1], [-1, -1, -1]]
    assert bool(torch.all(selected[0] < 4097))
    assert torch.equal(selected, topk(logits[:, :4097], 3, row_ends=row_lens))
    print(f"P2_PADDING unmasked={unmasked.tolist()} masked={selected.tolist()}")


@pytest.mark.parametrize("max_model_len,prefill_width", [
    (4096, 4096), (4097, 4100), (4098, 4100), (4099, 4100),
])
@pytest.mark.parametrize("total_seq_lens", [2, 32768])
def test_fp4_profile_accounts_for_aligned_prefill_and_unchanged_staging(
    max_model_len, prefill_width, total_seq_lens
):
    allocations = []

    def allocate(size, *, dtype, device):
        allocations.append((size, dtype, device))
        return object()

    hidden = torch.zeros(2, 128)
    out = torch.zeros(2, 3, dtype=torch.int32)
    fn = _reader(None, torch=SimpleNamespace(empty=allocate, uint8=torch.uint8,
                                            Tensor=torch.Tensor),
                 get_forward_context=lambda: SimpleNamespace(attn_metadata=None))
    returned = fn(hidden, "cache", None, None, None, None, 128, "ue8m0", 3,
                  128, max_model_len, total_seq_lens, out, skip_k_cache_insert=True)
    staging_bytes = 2 * ((max_model_len + 63) // 64 * 64) * (2 * 68 + 3 * 8 + 1)
    prefill_bytes = total_seq_lens * 68 + 2 * prefill_width * 4
    assert returned is out
    assert allocations == [(max(staging_bytes, prefill_bytes), torch.uint8, hidden.device)]
    assert allocations[0][0] >= staging_bytes
    assert allocations[0][0] >= prefill_bytes
    if total_seq_lens == 32768:
        old_prefill_bytes = total_seq_lens * 68 + 2 * max_model_len * 4
        assert prefill_bytes > staging_bytes
        assert allocations[0][0] - old_prefill_bytes == 2 * (prefill_width - max_model_len) * 4
    else:
        assert staging_bytes > prefill_bytes
    print(f"P2_PROFILE max_model_len={max_model_len} total_seq_lens={total_seq_lens} "
          f"prefill={prefill_bytes} staging={staging_bytes} allocated={allocations[0][0]}")
