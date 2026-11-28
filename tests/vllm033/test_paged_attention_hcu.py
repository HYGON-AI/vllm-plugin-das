"""Native writer/strict PA numerics with changed-input Graph replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from test_flash_attention_hcu import _make_impl, _reference_attention

pytestmark = [pytest.mark.hcu, pytest.mark.skipif(
    os.getenv("VLLM033_HCU_TEST") != "1", reason="requires prepared HCU runtime")]


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("cache_dtype", ("auto", "fp8_e4m3", "fp8_e5m2"))
@pytest.mark.parametrize("kvheads", (1, 4))
@pytest.mark.parametrize("dim", (64, 256))
@pytest.mark.parametrize("padding", (0, 64))
@torch.inference_mode()
def test_native_packed_decode_writer_strides_and_graph(layout, cache_dtype, kvheads, dim, padding):
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(947)
    batch, blocks, page, capacity, heads = 2, 6, 64, 192, 6 * kvheads
    dtype = torch.bfloat16
    storage_dtype = dtype if cache_dtype == "auto" else torch.uint8
    page_elements = kvheads * page * 2 * dim
    # Official allocator offsets are aligned. Keep a nonzero aligned offset:
    # the vendor's vectorized FP8 PA loads fault for byte-offset 13.
    offset = 64
    block_stride = page_elements + padding
    backing = torch.zeros(offset + blocks * block_stride + 17,
                          device="cuda", dtype=storage_dtype)
    inner = ((page * 2 * dim, 2 * dim, 1) if layout == "LBHNC"
             else (2 * dim, kvheads * 2 * dim, 1))
    packed = torch.as_strided(backing, (blocks, kvheads, page, 2 * dim),
                              (block_stride, *inner), offset)
    impl = _make_impl(layout, dim)
    impl.kv_cache_dtype = cache_dtype
    k, v = impl._get_kv_cache_views(packed)
    query = torch.randn(batch, 2 * heads, dim, device="cuda", dtype=dtype)[:, ::2]
    output_backing = torch.full_like(query.repeat_interleave(2, 1), 99)
    output = output_backing[:, ::2]
    table = torch.zeros(batch, 6, device="cuda", dtype=torch.int32)[:, ::2]
    lens = torch.zeros(2 * batch, device="cuda", dtype=torch.int32)[::2]
    cuq = torch.arange(2 * (batch + 1), device="cuda", dtype=torch.int32)[::2]
    cuq.copy_(torch.arange(batch + 1, device="cuda", dtype=torch.int32))
    slots = torch.empty(batch * capacity, device="cuda", dtype=torch.long)
    keys = torch.randn(batch * capacity, kvheads, dim, device="cuda", dtype=dtype)
    values = torch.randn_like(keys)
    ks = torch.ones((), device="cuda", dtype=torch.float32)
    vs = torch.ones_like(ks)
    layer = SimpleNamespace(_k_scale=ks, _v_scale=vs)
    fp8 = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}

    def produce(wave):
        lengths = ([77, 129], [130, 65], [64, 191])[wave]
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        table.copy_(torch.arange(blocks, device="cuda", dtype=torch.int32)
                    .roll(wave).reshape(batch, 3))
        positions = torch.arange(capacity, device="cuda")
        slots.copy_((table[:, positions // page].long() * page + positions % page).flatten())
        ks.fill_((1., .5, 2.)[wave]); vs.fill_((1., 2., .5)[wave])
        if wave:
            query.copy_(torch.randn_like(query)); keys.copy_(torch.randn_like(keys))
            values.copy_(torch.randn_like(values))
        ek, ev = keys, values
        if cache_dtype != "auto":
            ek = (keys.float() / ks).to(fp8[cache_dtype]).to(dtype) * ks
            ev = (values.float() / vs).to(fp8[cache_dtype]).to(dtype) * vs
        return torch.cat([_reference_attention(query[b:b+1],
            ek.reshape(batch, capacity, kvheads, dim)[b, :n],
            ev.reshape(batch, capacity, kvheads, dim)[b, :n])
            for b, n in enumerate(lengths)])

    def run():
        impl.do_kv_cache_update(layer, keys, values, packed, slots)
        extra = {} if cache_dtype == "auto" else dict(
            k_descale=ks.expand(batch, kvheads), v_descale=vs.expand(batch, kvheads))
        result = impl._run_flash_attention(q=query, k=k, v=v, out=output,
            cu_seqlens_q=cuq, seqused_k=lens, max_seqlen_q=1,
            max_seqlen_k=capacity, block_table=table, causal=True,
            softmax_scale=dim**-.5, **extra)
        assert result.data_ptr() == output.data_ptr()

    expected = produce(0)
    for _ in range(3): run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2)
    addresses = [t.data_ptr() for t in (packed, query, output, table, lens, cuq, ks, vs)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    previous = output.clone()
    for wave in (1, 2):
        expected = produce(wave)
        graph.replay(); torch.cuda.synchronize()
        observed = output.clone()
        run(); torch.cuda.synchronize()
        torch.testing.assert_close(observed, output, atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2)
        assert not torch.equal(output, previous)
        assert bool((output_backing[:, 1::2] == 99).all())
        assert bool((backing[:offset] == 0).all()) and bool((backing[-17:] == 0).all())
        assert addresses == [t.data_ptr() for t in (packed, query, output, table, lens, cuq, ks, vs)]
        previous = observed
