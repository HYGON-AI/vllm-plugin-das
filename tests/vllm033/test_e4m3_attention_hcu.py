"""Native E4M3 writer/reader, non-unit scales and changed-input Graph replay."""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from test_flash_attention_hcu import _make_impl, _make_packed_cache
from test_e5m2_scales_hcu import _reference

pytestmark = [pytest.mark.hcu, pytest.mark.skipif(
    os.getenv("VLLM033_HCU_TEST") != "1", reason="requires a prepared HCU node",
)]


@pytest.fixture
def poison_allocations():
    # Masked V lanes must have initialized backing storage. Otherwise Graph
    # pool reuse can make a dynamic-length test pass only by allocator luck.
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("dim", (64, 128, 192, 256))
@pytest.mark.parametrize("nq,chunked,wants_lse", (
    ((3, 11), False, True), ((1, 1), False, False),
    ((1, 1), False, True), ((3, 11), True, True), ((0, 14), True, True)))
@torch.inference_mode()
def test_native_e4m3_writer_reader_graph(
    monkeypatch, poison_allocations, layout, dtype, dim, nq, chunked, wants_lse,
):
    import vllm._C_stable_libtorch  # noqa: F401
    if chunked:
        from hcu_vllm_das.attention import e4m3_attention
        monkeypatch.setattr(e4m3_attention, "_PREFILL_SCRATCH_TARGET_BYTES", 1)

    torch.manual_seed(923)
    h, hk, page = 8, 2, 64
    nk = (77, 145)
    used_nk = nk
    max_q = max(nq)
    q = torch.randn(sum(nq), h, dim, device="cuda", dtype=dtype)
    keys = [torch.randn(n, hk, dim, device="cuda", dtype=dtype) for n in nk]
    values = [torch.randn_like(k) for k in keys]
    packed = _make_packed_cache(layout_name=layout, num_blocks=24, num_heads=hk,
        block_size=page, head_size=dim, device=q.device, dtype=torch.uint8, block_padding=64)
    impl = _make_impl(layout, dim)
    impl.kv_cache_dtype = "fp8_e4m3"
    kc, vc = impl._get_kv_cache_views(packed)
    wk, wv = impl._get_kv_cache_write_views(packed)
    table = torch.tensor([[9, 3, 17], [22, 1, 19]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor(nk, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, nq[0], sum(nq)], device="cuda", dtype=torch.int32)
    # Contiguous per-head scales exercise the native writer's scale stride.
    ks = torch.tensor([[.5, .75], [1.25, .625]], device="cuda")
    vs = torch.tensor([[1.25, .5], [.75, 1.5]], device="cuda")
    slots = [torch.empty(n, device="cuda", dtype=torch.int64) for n in nk]
    output = torch.empty_like(q)
    lse = []
    def run():
        for b, n in enumerate(nk):
            positions = torch.arange(n, device="cuda")
            slots[b].copy_(table[b][positions // page].long() * page + positions % page)
            impl.do_kv_cache_update(SimpleNamespace(_k_scale=ks[b], _v_scale=vs[b]),
                keys[b], values[b], packed, slots[b])
        result = impl._run_flash_attention(q=q, k=kc, v=vc, out=output,
            cu_seqlens_q=cu, seqused_k=lengths, max_seqlen_q=max_q,
            max_seqlen_k=max(nk), softmax_scale=dim ** -.5, causal=True,
            block_table=table, k_descale=ks, v_descale=vs,
            return_softmax_lse=wants_lse)
        assert (result[0] if isinstance(result, tuple) else result) is output
        if wants_lse:
            assert isinstance(result, tuple) and result[1].shape == (h, sum(nq))
            lse[:] = [result[1]]
    def check():
        if lse:
            assert bool(torch.isfinite(lse[0]).all())
        start = 0
        for b, n in enumerate(nq):
            quant_k = (keys[b].float() / ks[b, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            quant_v = (values[b].float() / vs[b, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            for written, expected in ((wk, quant_k), (wv, quant_v)):
                actual = written[slots[b] // page, slots[b] % page]
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            expected = _reference(q[start:start+n], quant_k[:used_nk[b]].float() * ks[b, :, None],
                                  quant_v[:used_nk[b]].float() * vs[b, :, None])
            torch.testing.assert_close(output[start:start+n].float(), expected, atol=.025, rtol=.025)
            if lse:
                # Independently check returned log-normalizers as well as output.
                scaled_k = quant_k[:used_nk[b]].float() * ks[b, :, None]
                scores = torch.einsum("qhd,khd->hqk", q[start:start+n].float(),
                    scaled_k.repeat_interleave(h // hk, dim=1)) / math.sqrt(dim)
                positions = torch.arange(used_nk[b], device=q.device)
                query_positions = used_nk[b] - n + torch.arange(n, device=q.device)
                scores.masked_fill_(positions[None, :] > query_positions[:, None], -float("inf"))
                torch.testing.assert_close(lse[0][:, start:start+n],
                    scores.logsumexp(-1), atol=.025, rtol=.025)
            start += n
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Eager calls below return a fresh auxiliary tensor. Replay writes the
    # LSE returned during capture, so retain that exact output across waves.
    graph_lse = tuple(lse)
    pointers = [x.data_ptr() for x in (q, packed, table, cu, lengths, ks, vs, output)]
    previous = output.clone()
    for wave in range(2):
        if chunked:
            # Keep addresses and static maxima, change actual query boundaries
            # and KV lengths. Include empty query rows in the 14-token case.
            nq = (7, 7) if wave == 0 else (max_q, 14 - max_q)
            used_nk = (64, 128) if wave == 0 else (76, 144)
            cu.copy_(torch.tensor([0, nq[0], sum(nq)], device=q.device, dtype=cu.dtype))
            lengths.copy_(torch.tensor(used_nk, device=q.device, dtype=lengths.dtype))
        q.copy_(torch.randn_like(q))
        for k, v in zip(keys, values):
            k.copy_(torch.randn_like(k)); v.copy_(torch.randn_like(v))
        ks.add_(.125); vs.add_(.125)
        table.copy_(table.flip(1))
        graph.replay(); torch.cuda.synchronize()
        lse[:] = graph_lse
        assert not torch.equal(output, previous)
        check()
        observed = output.clone()
        run(); torch.cuda.synchronize()
        assert torch.equal(output, observed)
        assert pointers == [x.data_ptr() for x in (q, packed, table, cu, lengths, ks, vs, output)]
        previous = observed


@torch.inference_mode()
def test_mixed_long_context_has_bounded_device_scratch(record_property):
    """The reported 48-GiB shape must run inside a small device budget."""
    import vllm._C_stable_libtorch  # noqa: F401

    batch, max_k, heads, kv_heads, dim, page = 256, 32768, 32, 8, 128, 64
    key = torch.zeros((max_k // page, page, kv_heads, dim),
                      device="cuda", dtype=torch.float8_e4m3fn)
    value = torch.full_like(key, .5)
    q = torch.zeros((batch + 1, heads, dim), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cu[1:].add_(1)  # one two-query prefill plus 255 single-query rows
    lengths = torch.ones(batch, device="cuda", dtype=torch.int32)
    lengths[0] = max_k
    table = torch.arange(max_k // page, device="cuda", dtype=torch.int32)
    table = table.expand(batch, -1).contiguous()
    impl = _make_impl("LBNHC", dim)
    impl.kv_cache_dtype = "fp8_e4m3"
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = impl._run_flash_attention(q=q, k=key, v=value, out=out,
        cu_seqlens_q=cu, seqused_k=lengths, max_seqlen_q=2,
        max_seqlen_k=max_k, softmax_scale=dim ** -.5, causal=True,
        block_table=table)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    record_property("peak_extra_allocated_bytes", peak)
    assert result is out
    torch.testing.assert_close(out, torch.full_like(out, .5), atol=0, rtol=0)
    # KV scratch is 192 MiB here. Allow vendor temporaries and query buffers
    # without permitting a return to the original batch*max-K reservation.
    assert peak < 768 * 1024**2
