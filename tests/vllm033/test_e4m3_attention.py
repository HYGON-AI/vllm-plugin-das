"""Portable E4M3 admission, native-writer and prefill composition contracts."""

from types import ModuleType
import sys

import pytest
import torch

from test_flash_attention import load_flash_attention, make_impl


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
def test_e4m3_reader_preserves_bytes_query_and_layout(monkeypatch, layout):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=layout)
    impl = make_impl(flash, kv_cache_dtype="fp8_e4m3")
    calls = []
    adapter = ModuleType("hcu_vllm_das.attention.e4m3_attention")
    adapter.run_e4m3_attention = lambda vendor, kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: object())
    key = torch.arange(128, dtype=torch.uint8).reshape(1, 1, 2, 64)
    query = torch.randn(1, 2, 64, dtype=torch.bfloat16)
    impl._run_flash_attention(q=query, k=key, v=key, max_seqlen_q=1)
    assert calls[0]["q"] is query
    assert calls[0]["k"].dtype == torch.float8_e4m3fn
    assert calls[0]["k"].data_ptr() == key.data_ptr()
    assert torch.equal(calls[0]["k"].view(torch.uint8), key)
    assert calls[0]["layout"] == ("bhsd" if layout == "LBHNC" else "bshd")
    assert not flash.HcuFlashAttentionBackend.supports_kv_cache_dtype("fp8_e4m3fnuz")
    assert not flash.HcuFlashAttentionBackend.supports_kv_cache_dtype("fp8")


@pytest.mark.parametrize("layout", ("bhsd", "bshd"))
@pytest.mark.parametrize("wants_lse", (False, True))
def test_prefill_uses_core_byte_gather_then_explicit_conversion(monkeypatch, layout, wants_lse):
    # Import before a separate test's Core seams are installed.
    from hcu_vllm_das.attention.e4m3_attention import _prefill

    native = ModuleType("vllm._custom_ops")
    gathers = []
    def gather(source, target, table, cumulative, batch):
        gathers.append((source.dtype, target.dtype, cumulative.tolist()))
        for seq in range(batch):
            start, end = cumulative[seq:seq + 2].tolist()
            for token in range(end - start):
                target[start + token].copy_(source[table[seq, token // 64], token % 64])
    native.cp_gather_cache = gather
    monkeypatch.setitem(sys.modules, native.__name__, native)
    key = (torch.arange(2 * 64 * 2 * 64).reshape(2, 64, 2, 64).float() % 15).to(torch.float8_e4m3fn)
    if layout == "bhsd":
        key = key.transpose(1, 2)
    query = torch.randn(3, 4, 64, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    table = torch.tensor([[1], [0]], dtype=torch.int32)
    def vendor(**kwargs):
        assert kwargs["q"] is query
        assert "block_table" not in kwargs and "layout" not in kwargs
        assert kwargs["k"].dtype == query.dtype
        assert kwargs["cu_seqlens_k"].tolist() == [0, 3, 5]
        # The final sequence's rounded tile must also be initialized. A
        # masked NaN in V can still propagate through the vendor's PV tile.
        assert kwargs["k"].shape[0] >= 3 + 256
        assert not bool(kwargs["k"][5:].count_nonzero())
        assert not bool(kwargs["v"][5:].count_nonzero())
        expected_key = key.transpose(1, 2) if layout == "bhsd" else key
        expected = torch.cat((expected_key[1, :3].float(), expected_key[0, :2].float()))
        torch.testing.assert_close(kwargs["k"][:5].float(), expected, atol=0, rtol=0)
        result = torch.full_like(query, 7)
        return (result, "lse", "aux") if kwargs.get("return_attn_probs") else result
    result = _prefill(vendor, dict(q=query, k=key, v=key, out=output, layout=layout,
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        seqused_k=lengths, max_seqlen_k=3, block_table=table,
        return_softmax_lse=wants_lse))
    assert len(gathers) == 4
    assert all(a == b == torch.float8_e4m3fn and c == [0, 3, 5] for a, b, c in gathers)
    assert (result[0] if wants_lse else result) is output
    assert bool((output == 7).all())
    if wants_lse:
        assert result[1:] == ("lse", "aux")


def test_mixed_long_context_workspace_does_not_expand_every_sequence(monkeypatch):
    """Exercise the reported 48-GiB shape without allocating device memory."""
    from hcu_vllm_das.attention.e4m3_attention import _prefill

    native = ModuleType("vllm._custom_ops")
    capacities = []
    native.cp_gather_cache = lambda source, target, table, cumulative, batch: capacities.append(target.shape[0])
    monkeypatch.setitem(sys.modules, native.__name__, native)
    key = torch.empty((512, 64, 8, 128), device="meta", dtype=torch.float8_e4m3fn)
    q = torch.empty((257, 32, 128), device="meta", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    result = _prefill(lambda **kw: kw["out"], dict(q=q, k=key, v=key, out=out,
        layout="bshd", cu_seqlens_q=torch.empty(257, device="meta", dtype=torch.int32),
        seqused_k=torch.empty(256, device="meta", dtype=torch.int32),
        max_seqlen_q=2, max_seqlen_k=32768,
        block_table=torch.empty((256, 512), device="meta", dtype=torch.int32)))
    assert result is out
    assert capacities and max(capacities) <= 32768
    # Two FP8 gathers plus two BF16 conversions: at most one context here.
    assert max(capacities) * 8 * 128 * 6 <= 192 * 1024**2


@pytest.mark.parametrize("wants_lse", (False, True))
@pytest.mark.parametrize("group_size", (1, 2))
def test_chunked_queries_and_lse_keep_empty_rows_and_padding(monkeypatch, wants_lse, group_size):
    from hcu_vllm_das.attention import e4m3_attention as attention

    monkeypatch.setattr(attention, "_PREFILL_SCRATCH_TARGET_BYTES", group_size * 256 * 4 * 6)
    native = ModuleType("vllm._custom_ops")
    def gather(source, target, table, cumulative, batch):
        for b in range(batch):
            start, end = cumulative[b:b + 2].tolist()
            target[start:end].copy_(source[table[b, 0], :end - start])
    native.cp_gather_cache = gather
    monkeypatch.setitem(sys.modules, native.__name__, native)
    key = (torch.arange(3 * 64 * 4).reshape(3, 64, 1, 4).float() % 13).to(torch.float8_e4m3fn)
    value = (key.float() * .5).to(torch.float8_e4m3fn)
    q = torch.randn(5, 2, 4, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    def reference(query, k, v):
        scores = torch.einsum("qhd,khd->hqk", query.float(), k.float().expand(-1, 2, -1)) * .5
        q_positions = k.shape[0] - query.shape[0] + torch.arange(query.shape[0])
        scores.masked_fill_(torch.arange(k.shape[0])[None, :] > q_positions[:, None], -torch.inf)
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float().expand(-1, 2, -1)), scores.logsumexp(-1)
    calls = []
    def vendor(**kw):
        calls.append(kw["cu_seqlens_q"].tolist())
        out = torch.full_like(kw["q"], torch.nan)
        lse = torch.full((2, kw["q"].shape[0]), torch.nan)
        assert kw["cu_seqlens_q"][0] == 0
        for b in range(len(kw["cu_seqlens_q"]) - 1):
            qs, qe = kw["cu_seqlens_q"][b:b + 2].tolist()
            ks, ke = kw["cu_seqlens_k"][b:b + 2].tolist()
            if qe > qs:
                y, normalizer = reference(kw["q"][qs:qe], kw["k"][ks:ke], kw["v"][ks:ke])
                out[qs:qe].copy_(y)
                lse[:, qs:qe].copy_(normalizer)
        return (out, lse, torch.empty(0)) if kw.get("return_attn_probs") else out
    for counts, lengths in [([1, 3, 0], [2, 4, 1]), ([2, 0, 2], [3, 1, 3])]:
        cuq = torch.tensor([0, counts[0], counts[0] + counts[1], sum(counts)], dtype=torch.int32)
        output.fill_(-99)
        result = attention._prefill(vendor, dict(q=q, k=key, v=value, out=output,
            layout="bshd", cu_seqlens_q=cuq,
            seqused_k=torch.tensor(lengths, dtype=torch.int32), max_seqlen_q=3,
            max_seqlen_k=4, block_table=torch.arange(3, dtype=torch.int32)[:, None],
            return_softmax_lse=wants_lse))
        assert (result[0] if wants_lse else result) is output
        for b, count in enumerate(counts):
            if count:
                start, end = cuq[b:b + 2].tolist()
                expected, normalizer = reference(q[start:end], key[b, :lengths[b]], value[b, :lengths[b]])
                torch.testing.assert_close(output[start:end], expected.to(q.dtype), atol=0, rtol=0)
                if wants_lse:
                    torch.testing.assert_close(result[1][:, start:end], normalizer)
        assert bool((output[sum(counts):] == -99).all())
    assert len(calls) == (6 if group_size == 1 else 4)
    assert all(2 <= len(c) <= group_size + 1 and c[0] == 0 for c in calls)
