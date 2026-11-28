"""Strict PA consumes dense page views without copying the shared KV cache."""

import pytest
import torch


@pytest.mark.parametrize("layout", ("bhsd", "bshd"))
@pytest.mark.parametrize("gap", (2, 3))
def test_page_remapping_preserves_offsets_aliases_and_page_contents(layout, gap):
    from hcu_vllm_das.attention.paged_attention import run_paged_attention

    shape = (4, 2, 64, 8) if layout == "bhsd" else (4, 64, 2, 8)
    page = 2 * 64 * 8
    backing = torch.arange(23 + 4 * gap * page + 17, dtype=torch.float32)
    dense = torch.empty(shape).stride()
    k = torch.as_strided(backing, shape, (gap * page, *dense[1:]), 23)
    v = torch.as_strided(backing, shape, (gap * page, *dense[1:]), 23 + page)
    table = torch.tensor([[3, 0, 2], [1, 2, 0]], dtype=torch.int32)
    original = table.clone()
    q = torch.randn(2, 8, 8)[:, ::2]
    output_backing = torch.full((2, 8, 8), -9.)
    output = output_backing[:, ::2]
    lengths = torch.tensor([70, 0, 90, 0], dtype=torch.int32)[::2]
    def vendor(**kw):
        for name in ("q", "k", "v", "out", "block_table", "seqused_k"):
            assert kw[name].is_contiguous(), name
        assert kw["k"].data_ptr() == k.data_ptr()
        assert kw["v"].data_ptr() == v.data_ptr()
        for b in range(4):
            torch.testing.assert_close(kw["k"][gap * b], k[b])
            torch.testing.assert_close(kw["v"][gap * b], v[b])
        torch.testing.assert_close(kw["block_table"], original * gap)
        kw["out"].fill_(7)
        return kw["out"]
    result = run_paged_attention(vendor, dict(q=q, k=k, v=v, out=output,
        block_table=table, seqused_k=lengths, max_seqlen_q=1, layout=layout))
    assert result is output
    assert bool((output == 7).all())
    assert bool((output_backing[:, 1::2] == -9).all())
    torch.testing.assert_close(table, original)


def test_dense_pages_keep_original_table_and_output():
    from hcu_vllm_das.attention.paged_attention import run_paged_attention

    key = torch.empty(3, 64, 2, 8)
    output = torch.empty(2, 4, 8)
    table = torch.zeros((2, 1), dtype=torch.int32)
    def vendor(**kw):
        assert kw["k"] is key and kw["v"] is key
        assert kw["out"] is output and kw["block_table"] is table
        return output
    assert run_paged_attention(vendor, dict(q=output, k=key, v=key, out=output,
        block_table=table, seqused_k=torch.ones(2, dtype=torch.int32),
        max_seqlen_q=1, layout="bshd")) is output


def test_prefill_is_forwarded_without_reindexing():
    from hcu_vllm_das.attention.paged_attention import run_paged_attention

    arguments = dict(max_seqlen_q=2, block_table=object())
    assert run_paged_attention(lambda **kw: kw, arguments) == arguments


def test_incompatible_page_strides_are_rejected_without_cache_copy():
    from hcu_vllm_das.attention.paged_attention import run_paged_attention

    k = torch.empty(8, 64, 2, 8)[::2]
    v = torch.empty(12, 64, 2, 8)[::3]
    with pytest.raises(ValueError, match="same page stride"):
        run_paged_attention(lambda **kw: pytest.fail("must reject"),
            dict(q=torch.empty(1, 4, 8), k=k, v=v, max_seqlen_q=1,
                 block_table=torch.zeros(1, 1, dtype=torch.int32), layout="bshd"))


@pytest.mark.parametrize("provide_out", (False, True))
def test_subpage_padding_uses_bounded_gather_without_changing_cache(monkeypatch, provide_out):
    from hcu_vllm_das.attention import e4m3_attention
    from hcu_vllm_das.attention.paged_attention import run_paged_attention

    cache = torch.empty_strided((3, 64, 2, 8), (2 * 64 * 2 * 8 + 64, 16, 8, 1))
    output = torch.empty(2, 4, 8)
    table = torch.tensor([[2, 0], [1, 0]], dtype=torch.int32)[:, ::2]
    def bounded(vendor, kw):
        assert kw["k"] is cache and kw["v"] is cache
        assert kw["block_table"].is_contiguous()
        torch.testing.assert_close(kw["block_table"], table)
        kw["out"].fill_(3)
        return kw["out"]
    monkeypatch.setattr(e4m3_attention, "_prefill", bounded)
    arguments = dict(q=output, k=cache, v=cache, max_seqlen_q=1, block_table=table)
    if provide_out:
        arguments["out"] = output
    result = run_paged_attention(object(), arguments)
    if provide_out:
        assert result is output
    assert bool((result == 3).all())
