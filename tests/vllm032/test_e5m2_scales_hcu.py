"""Non-unit-scale prefill/decode through the real vendor ABI and Graph."""

import math
import os

import pytest
import torch

from test_flash_attention_hcu import _make_impl, _make_packed_cache

pytestmark = pytest.mark.skipif(
    os.getenv("VLLM032_HCU_TEST") != "1" or not torch.cuda.is_available(),
    reason="set VLLM032_HCU_TEST=1 on a prepared HCU node",
)


def _reference(q, k, v):
    # Independent FP32 GQA with bottom-right causal alignment.
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(q.shape[-1])
    positions = torch.arange(k.shape[0], device=q.device)
    q_positions = k.shape[0] - q.shape[0] + torch.arange(q.shape[0], device=q.device)
    scores.masked_fill_(positions[None, :] > q_positions[:, None], -float("inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("nq", ((3, 11), (1, 1)))
@pytest.mark.parametrize("scale_multiplier", (1.0, 0.001))
@torch.inference_mode()
def test_nonunit_attention_changed_graph_inputs(layout, dtype, nq, scale_multiplier):
    import vllm._C_stable_libtorch  # noqa: F401
    from hcu_vllm_das.attention.e5m2_cache import reshape_and_cache_e5m2

    torch.manual_seed(321)
    device = "cuda"
    h, hk, d, page = 8, 2, 64, 64
    nk = (77, 145)
    query = torch.randn(sum(nq), h, d, device=device, dtype=dtype)
    keys = [torch.randn(n, hk, d, device=device, dtype=dtype) for n in nk]
    values = [torch.randn_like(k) for k in keys]
    packed = _make_packed_cache(
        layout_name=layout, num_blocks=24, num_heads=hk, block_size=page,
        head_size=d, device=torch.device(device), dtype=torch.uint8,
        block_padding=64,
    )
    impl = _make_impl(layout, d)
    impl.kv_cache_dtype = "fp8_e5m2"
    key_cache, value_cache = impl._get_kv_cache_views(packed)
    write_k, write_v = impl._get_kv_cache_write_views(packed)
    table = torch.tensor([[9, 3, 17], [22, 1, 19]], device=device, dtype=torch.int32)
    lengths = torch.tensor(nk, device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, nq[0], sum(nq)], device=device, dtype=torch.int32)
    # Each sequence has a distinct scalar, expanded with zero head stride just
    # as vLLM expands layer scale tensors; replay changes values at the same address.
    ks = (torch.tensor([[0.5], [0.75]], device=device) * scale_multiplier).expand(2, hk)
    vs = (torch.tensor([[1.25], [1.5]], device=device) * scale_multiplier).expand(2, hk)
    out = torch.empty_like(query)
    slots = [torch.empty(n, device=device, dtype=torch.int64) for n in nk]
    lse = []

    def run():
        for b, n in enumerate(nk):
            tokens = torch.arange(n, device=device)
            slots[b].copy_(table[b][tokens // page].long() * page + tokens % page)
            reshape_and_cache_e5m2(
                keys[b], values[b], write_k, write_v, slots[b], ks[b, 0], vs[b, 0]
            )
        result = impl._run_flash_attention(
            q=query, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=cu_q, seqused_k=lengths,
            max_seqlen_q=max(nq), max_seqlen_k=max(nk),
            softmax_scale=d ** -0.5, causal=True, block_table=table,
            k_descale=ks, v_descale=vs, return_softmax_lse=max(nq) > 1,
        )
        if max(nq) > 1:
            assert isinstance(result, tuple) and result[0] is out
            lse[:] = [result[1]]
        else:
            assert result is out

    def check():
        start = 0
        for b, n in enumerate(nq):
            k = (keys[b].float() / ks[b, :, None]).to(torch.float8_e5m2).float()
            v = (values[b].float() / vs[b, :, None]).to(torch.float8_e5m2).float()
            expected = _reference(
                query[start:start+n], k * ks[b, :, None], v * vs[b, :, None]
            )
            torch.testing.assert_close(out[start:start+n].float(), expected, atol=0.025, rtol=0.025)
            start += n
        if lse:
            assert lse[0].shape == (h, sum(nq))
            assert torch.isfinite(lse[0]).all()

    saved_query = query.clone()
    for _ in range(3):
        run()
    torch.testing.assert_close(query, saved_query, atol=0, rtol=0)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for wave in range(3):
        query.copy_(torch.randn_like(query))
        for key, value in zip(keys, values):
            key.copy_(torch.randn_like(key))
            value.copy_(torch.randn_like(value))
        ks[:, 0].copy_(torch.tensor([0.625 + wave * 0.125, 1.125 + wave * 0.125], device=device) * scale_multiplier)
        vs[:, 0].copy_(torch.tensor([1.375 + wave * 0.125, 0.875 + wave * 0.125], device=device) * scale_multiplier)
        table.copy_(table.flip(1))
        saved_query = query.clone()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(query, saved_query, atol=0, rtol=0)
        check()
