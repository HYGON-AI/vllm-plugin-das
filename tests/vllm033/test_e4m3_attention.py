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
