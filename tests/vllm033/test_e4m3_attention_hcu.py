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


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("dim", (64, 128, 192, 256))
@pytest.mark.parametrize("nq", ((3, 11), (1, 1)))
@torch.inference_mode()
def test_native_e4m3_writer_reader_graph(layout, dtype, dim, nq):
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(923)
    h, hk, page = 8, 2, 64
    nk = (77, 145)
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
            cu_seqlens_q=cu, seqused_k=lengths, max_seqlen_q=max(nq),
            max_seqlen_k=max(nk), softmax_scale=dim ** -.5, causal=True,
            block_table=table, k_descale=ks, v_descale=vs,
            return_softmax_lse=max(nq) > 1)
        assert (result[0] if isinstance(result, tuple) else result) is output
        if max(nq) > 1:
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
            expected = _reference(q[start:start+n], quant_k.float() * ks[b, :, None],
                                  quant_v.float() * vs[b, :, None])
            torch.testing.assert_close(output[start:start+n].float(), expected, atol=.025, rtol=.025)
            if lse:
                # Independently check returned log-normalizers as well as output.
                scaled_k = quant_k.float() * ks[b, :, None]
                scores = torch.einsum("qhd,khd->hqk", q[start:start+n].float(),
                    scaled_k.repeat_interleave(h // hk, dim=1)) / math.sqrt(dim)
                positions = torch.arange(nk[b], device=q.device)
                query_positions = nk[b] - n + torch.arange(n, device=q.device)
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
