"""Independent masked references for native writer, ragged windows and Graph."""
import os
from types import SimpleNamespace

import pytest
import torch

from test_flash_attention_hcu import _make_impl, _make_packed_cache

pytestmark = [pytest.mark.hcu, pytest.mark.skipif(
    os.getenv("VLLM033_HCU_TEST") != "1",
    reason="set VLLM033_HCU_TEST=1 on a prepared HCU node",
)]


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("kv_dtype,k_scale,v_scale", (
    ("bfloat16", 1.0, 1.0), ("float16", 1.0, 1.0),
    ("fp8_e4m3", 0.5, 0.25), ("fp8_e4m3", 0.7, 0.3),
    ("fp8_e5m2", 0.5, 0.25), ("fp8_e5m2", 0.7, 0.3),
))
@pytest.mark.parametrize("softcap", (0.0, 50.0))
@pytest.mark.parametrize("window", (1, 17, 64, 4096))
@torch.inference_mode()
def test_sliding_native_writer_forward_ragged_changed_graph(
    layout, kv_dtype, k_scale, v_scale, softcap, window,
):
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(417)
    dtype = torch.float16 if kv_dtype == "float16" else torch.bfloat16
    fp8 = kv_dtype.startswith("fp8_")
    storage_dtype = torch.uint8 if fp8 else dtype
    quant_dtype = torch.float8_e4m3fn if kv_dtype == "fp8_e4m3" else torch.float8_e5m2
    batch, capacity, heads, kvheads, dim = 4, 192, 8, 4, 256
    query = torch.empty(6, heads, dim, device="cuda", dtype=dtype)
    keys = torch.empty(batch * capacity, kvheads, dim, device="cuda", dtype=dtype)
    values = torch.empty_like(keys)
    packed = _make_packed_cache(layout_name=layout, num_blocks=15, num_heads=kvheads,
        block_size=64, head_size=dim, device=query.device, dtype=storage_dtype,
        block_padding=128)
    impl = _make_impl(layout, dim)
    impl.num_heads, impl.num_kv_heads = heads, kvheads
    impl.kv_cache_dtype, impl.logits_soft_cap = kv_dtype, softcap
    impl.sliding_window = (window - 1, 0)
    impl.alibi_slopes = None
    table = torch.arange(12, device="cuda", dtype=torch.int32).reshape(batch, 3)
    initial_table = table.clone()
    cuq = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(batch, device="cuda", dtype=torch.int32)
    slots = torch.empty(batch * capacity, device="cuda", dtype=torch.long)
    output = torch.full((8, heads, dim), 99, device="cuda", dtype=dtype)
    kscale = torch.tensor(k_scale, device="cuda", dtype=torch.float32)
    vscale = torch.tensor(v_scale, device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(_k_scale=kscale, _v_scale=vscale)
    meta = SimpleNamespace(num_actual_tokens=6, causal=True, query_start_loc=cuq,
        max_query_len=3, seq_lens=lengths, max_seq_len=capacity, block_table=table,
        mm_prefix_query_range_tensor=None, rswa_prefix_lens=None, dcp_context_kv_lens=None)
    impl.scale = dim ** -0.5

    def produce(wave):
        qlens = ((1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0))[wave]
        lens = ((75, 129, 0, 66), (65, 0, 130, 128), (129, 65, 127, 0))[wave]
        offsets = [0]
        for n in qlens: offsets.append(offsets[-1] + n)
        cuq.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        table.copy_(initial_table.roll(wave, 1))
        positions = torch.arange(capacity, device="cuda")
        addresses = table[:, positions // 64].long() * 64 + positions % 64
        slots.copy_(torch.where(positions[None, :] < lengths[:, None], addresses, -1).flatten())
        query.copy_(3 * torch.randn_like(query))
        keys.copy_(3 * torch.randn_like(keys))
        values.copy_(torch.randn_like(values))
        # Reference is independent of paged storage, its writer and Core kernels.
        kr, vr = keys.float(), values.float()
        if fp8:
            kr = (kr / kscale).to(quant_dtype).float() * kscale
            vr = (vr / vscale).to(quant_dtype).float() * vscale
        kr = kr.reshape(batch, capacity, kvheads, dim)
        vr = vr.reshape_as(kr)
        expected = []
        for b, (nq, nk) in enumerate(zip(qlens, lens, strict=True)):
            if not nq: continue
            q = query[offsets[b]:offsets[b+1]].float()
            k = kr[b, :nk].repeat_interleave(heads // kvheads, dim=1)
            v = vr[b, :nk].repeat_interleave(heads // kvheads, dim=1)
            logits = torch.einsum("qhd,khd->qhk", q, k) * dim ** -0.5
            if softcap: logits = softcap * torch.tanh(logits / softcap)
            qpos = nk - nq + torch.arange(nq, device="cuda")
            kpos = torch.arange(nk, device="cuda")
            visible = ((kpos[None, :] <= qpos[:, None])
                       & (kpos[None, :] > qpos[:, None] - window))
            logits.masked_fill_(~visible[:, None, :], float("-inf"))
            expected.append(torch.einsum("qhk,khd->qhd", logits.softmax(-1), v))
        return torch.cat(expected).to(dtype)

    def run():
        impl.do_kv_cache_update(layer, keys, values, packed, slots)
        assert impl.forward(layer, query, keys, values, packed, meta, output) is output

    expected = produce(0)
    for _ in range(3): run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:6], expected, atol=1.6e-2, rtol=1.6e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    buffers = (query, keys, values, packed, cuq, lengths, table, slots, output)
    addresses = [t.data_ptr() for t in buffers]
    previous = output[:6].clone()
    for wave in (1, 2):
        expected = produce(wave)
        graph.replay(); torch.cuda.synchronize()
        observed = output[:6].clone()
        run(); torch.cuda.synchronize()
        assert torch.equal(observed, output[:6])
        assert not torch.equal(observed, previous)
        torch.testing.assert_close(observed, expected, atol=1.6e-2, rtol=1.6e-2)
        assert bool(torch.all(output[6:] == 99))
        assert addresses == [t.data_ptr() for t in buffers]
        previous = observed
