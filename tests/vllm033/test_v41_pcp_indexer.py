"""CPU oracle for the eager PCP FP8 cache scorer."""
import pytest
import torch
from hcu_vllm_das import v41_indexer as candidate
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as rocm
B, D, HEADS = 16, 128, 2


@pytest.fixture(autouse=True)
def interpret_gpu_score_on_cpu(monkeypatch):
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction
    from hcu_vllm_das import v41_indexer_gpu as gpu
    # tl.sum and its nested helpers were imported before this fixture.
    monkeypatch.setattr(JITFunction, "__call__",
        lambda self, *args, **kwargs: InterpretedFunction(self.fn)(*args, **kwargs))
    monkeypatch.setattr(gpu, "_paged_scores", InterpretedFunction(gpu._paged_scores.fn))
    yield
    assert not torch.cuda.is_initialized()

def pack(seed):
    g=torch.Generator().manual_seed(seed)
    logical=torch.randint(-4,5,(5,B,D),generator=g).to(rocm.FP8_DTYPE)
    scales=torch.randint(1,8,(5,B),generator=g).float()/4
    raw=torch.zeros((5,B*(D+4)),dtype=torch.uint8)
    # Independent explicit writer offset formula, not the candidate reshape.
    cols=torch.arange(D)
    for token in range(B):
        offsets=(token//16)*D*16+(cols//16)*256+(token%16)*16+cols%16
        raw[:,offsets]=logical[:,token].view(torch.uint8)
    raw[:,B*D:]=scales.contiguous().view(torch.uint8).reshape(5,B*4)
    return raw.view(5,B,D+4),logical,scales


def reference_decode(q,logical,scale,weights,lens,tables,maxlen):
    reqs,nextn,heads,dim=q.shape
    out=torch.full((reqs*nextn,maxlen),-float('inf'))
    limits=lens.reshape(reqs,-1)
    if lens.ndim==1:
        limits=(limits-nextn+torch.arange(nextn)[None,:]+1).clamp_min(0)
    for req in range(reqs):
        for j in range(nextn):
            end=int(limits[req,j]);row=req*nextn+j
            for pos in range(end):
                page=int(tables[req,pos//B]);slot=pos%B
                dots=(q[req,j].float()*logical[page,slot].float()).sum(-1)
                out[row,pos]=(dots.relu()*weights[row]).sum()*scale[page,slot]
    return out,limits


def test_decode_200():
    for seed in range(200):
        cache,logical,scales=pack(seed);g=torch.Generator().manual_seed(seed+999)
        n=1;q=torch.randint(-3,4,(2,n,HEADS,D),generator=g).to(rocm.FP8_DTYPE)
        weights=torch.randint(1,8,(2*n,HEADS),generator=g).float()/8
        lens=torch.tensor([seed%24+1,seed%17+3],dtype=torch.int32)
        if seed%2:lens=(lens[:,None]-n+torch.arange(n)+1).clamp_min(0)
        tables=torch.tensor([[3,1],[4,2]],dtype=torch.int32)
        actual=candidate.decode_logits(q,cache,weights,lens,tables,None,32)
        expected,_=reference_decode(q,logical,scales,weights,lens,tables,32)
        assert torch.equal(torch.isneginf(actual),torch.isneginf(expected)),seed
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    r,_,_=pack(6)
    bad=r[...,:-1];pytest.raises(ValueError, candidate.decode_logits, q,bad,weights,lens,tables,None,32)
    # One correct packed token page uses NORMAL storage, but must retain tail scales.
    raw=torch.zeros((1,1,D+4),dtype=torch.uint8)
    raw[0,0,:D]=torch.ones(D,dtype=rocm.FP8_DTYPE).view(torch.uint8)
    raw[0,0,D:]=torch.tensor([.5]).view(torch.uint8)
    got=candidate.decode_logits(torch.ones(1,1,HEADS,D,dtype=rocm.FP8_DTYPE),raw,torch.ones(1,HEADS),torch.tensor([1]),torch.tensor([[0]]),None,1)
    assert got.item()==D


def test_reuse_native_indexer_without_changing_core_globals():
    import inspect
    core = inspect.unwrap(candidate.native.rocm_aiter_sparse_attn_indexer)
    assert candidate._indexer.__code__ is core.__code__
    assert candidate._indexer.__globals__["rocm_fp8_paged_mqa_logits"] is candidate.decode_logits
    assert core.__globals__["rocm_fp8_paged_mqa_logits"] is candidate.native.rocm_fp8_paged_mqa_logits
    assert candidate._indexer.__globals__["rocm_fp8_mqa_logits"] is candidate.prefill_logits
    assert core.__globals__["rocm_fp8_mqa_logits"] is candidate.native.rocm_fp8_mqa_logits
    for name in ["cp_gather_indexer_k_quant_cache_triton",
                 "current_workspace_manager"]:
        assert candidate._indexer.__globals__[name] is core.__globals__[name]


def test_execute_core_decode_with_owned_score_callback(monkeypatch):
    from types import SimpleNamespace as NS
    from vllm.config import CUDAGraphMode
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata, DeepSeekV32IndexerDecodeMetadata)
    cache, logical, scales = pack(12)
    before = cache.clone()
    q = torch.ones((2, HEADS, D), dtype=rocm.FP8_DTYPE)
    weights = torch.ones(2, HEADS)
    lens = torch.tensor([6, 9], dtype=torch.int32)
    tables = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    decode = DeepSeekV32IndexerDecodeMetadata(
        tables, lens, torch.ones(2, dtype=torch.int32), False, torch.empty(0))
    meta = DeepseekV32IndexerMetadata(
        lens, 9, torch.zeros(2, dtype=torch.int64), 2, 2, 0, 0, decode=decode)
    context = NS(attn_metadata={"k": meta}, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setitem(candidate._indexer.__globals__, "get_forward_context", lambda: context)
    monkeypatch.setattr(rocm_aiter_ops, "is_indexer_top_k_supported", lambda **kw: False)
    calls = []
    def topk(logits, next_n, seq_lens, output, *args):
        calls.append(logits.clone())
        output.fill_(7)
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_decode", topk, raising=False)
    output = torch.zeros(2, 8, dtype=torch.int32)
    layer = NS(k_cache=NS(prefix="k", kv_cache=cache), topk_indices_buffer=output,
        skip_k_cache_insert=True, use_fp4_cache=False, quant_block_size=128,
        scale_fmt="ue8m0", topk_tokens=8, head_dim=D, max_model_len=32,
        max_total_seq_len=64, compress_ratio=1, candidate_blocks=None,
        candidate_block_size=0, candidate_write=False)
    assert candidate.indexer(layer, torch.zeros(2, 4), q, None, weights) is output
    expected, _ = reference_decode(q[:, None], logical, scales, weights, lens, tables, 32)
    torch.testing.assert_close(calls[0], expected, atol=0, rtol=0)
    assert torch.equal(cache, before) and (output == 7).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("scale_column", [False, True])
def test_prefill_chunking_preserves_core_scores_and_masks(monkeypatch, scale_column):
    generator = torch.Generator().manual_seed(42)
    rows, heads, keys, dim = 7, 3, 11, 16
    q = torch.randint(-3, 4, (rows, heads, dim), generator=generator).to(rocm.FP8_DTYPE)
    k = torch.randint(-3, 4, (keys, dim), generator=generator).to(rocm.FP8_DTYPE)
    scales = torch.rand(keys, generator=generator) + .1
    if scale_column:
        scales = scales[:, None]
    weights = torch.rand(rows, heads, generator=generator)
    starts = torch.tensor([0, 1, 2, 0, 4, 3, 8], dtype=torch.int32)
    ends = torch.tensor([1, 4, 7, 9, 11, 6, 8], dtype=torch.int32)
    expected = rocm.fp8_mqa_logits_torch(q, (k, scales), weights, starts, ends)
    calls = []
    def core(q_slice, kv, w_slice, lo, hi):
        calls.append(q_slice.shape[0])
        return rocm.fp8_mqa_logits_torch(q_slice, kv, w_slice, lo, hi)
    monkeypatch.setattr(candidate.native, "rocm_fp8_mqa_logits", core)
    monkeypatch.setattr(candidate, "_PREFILL_SCORER", None)
    monkeypatch.setattr(candidate, "_PREFILL_SCORE_WORKSPACE_BYTES", 2 * heads * keys * 4)
    actual = candidate.prefill_logits(q, (k, scales), weights, starts, ends)
    assert calls == [2, 2, 2, 1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls.clear()
    monkeypatch.setattr(candidate, "_PREFILL_SCORE_WORKSPACE_BYTES", rows * heads * keys * 4)
    torch.testing.assert_close(candidate.prefill_logits(q, (k, scales), weights, starts, ends), expected)
    assert calls == [rows]


def test_prefill_uses_triton_scorer_without_torch_chunks(monkeypatch):
    rows, heads, keys, dim = 5, 4, 9, 16
    q = torch.zeros((rows, heads, dim), dtype=rocm.FP8_DTYPE)
    k = torch.zeros((keys, dim), dtype=rocm.FP8_DTYPE)
    scales = torch.ones((keys, 1))
    weights = torch.ones((rows, heads))
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.arange(1, rows + 1, dtype=torch.int32)
    sentinel = torch.empty((rows, keys))
    calls = []

    def scorer(*args):
        calls.append(args)
        return sentinel

    def core(*args):
        raise AssertionError("Torch fallback must not run when the scorer applies")

    monkeypatch.setattr(candidate, "_PREFILL_SCORER", scorer)
    monkeypatch.setattr(candidate.native, "rocm_fp8_mqa_logits", core)
    monkeypatch.setattr(candidate, "_PREFILL_SCORE_WORKSPACE_BYTES", 1)
    assert candidate.prefill_logits(q, (k, scales), weights, starts, ends) is sentinel
    (args,) = calls
    assert args[0] is q and args[1] is k and args[3] is weights
    assert args[2].shape == (keys,) and args[2].data_ptr() == scales.data_ptr()
    assert args[4] is starts and args[5] is ends
