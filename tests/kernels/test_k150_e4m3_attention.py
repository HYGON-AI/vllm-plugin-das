# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""K150 E4M3 cache reader against an independent FP32 attention oracle."""
from types import SimpleNamespace

import pytest
import torch

from tests.fixtures.flash_attention import load_hcu_flash_attention_module


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("qlen", [1, 4, 32, 128])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.25, 0.5)])
def test_k150_e4m3_attention(monkeypatch, qlen, layout, scales):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx92a"):
        pytest.skip("requires K150")
    module = load_hcu_flash_attention_module(monkeypatch)
    monkeypatch.setattr(module, "get_current_vllm_config_or_none", lambda: None)
    impl = module.FlashAttentionImpl(
        num_heads=24, head_size=256, scale=0.0625, num_kv_heads=4,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="fp8_e4m3",
    )
    torch.manual_seed(17)
    q = torch.randn(qlen, 24, 256, device="cuda", dtype=torch.bfloat16)
    # Retain token-order values for the oracle; physical pages are reversed.
    k = torch.randn(256, 4, 256, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(256, 4, 256, device="cuda").to(torch.float8_e4m3fn)
    cache = torch.empty((4, 2, 4, 64, 256) if layout == "HND" else (4, 2, 64, 4, 256), device="cuda", dtype=torch.uint8)
    logical = cache.transpose(2, 3) if layout == "HND" else cache
    logical[:, 0].copy_(k.view(torch.uint8).reshape(4, 64, 4, 256).flip(0))
    logical[:, 1].copy_(v.view(torch.uint8).reshape(4, 64, 4, 256).flip(0))
    layer = SimpleNamespace(_k_scale=torch.tensor(scales[0], device="cuda"), _v_scale=torch.tensor(scales[1], device="cuda"))
    metadata = SimpleNamespace(
        num_actual_tokens=qlen, query_start_loc=torch.tensor([0, qlen], device="cuda", dtype=torch.int32),
        max_query_len=qlen, seq_lens=torch.tensor([200], device="cuda", dtype=torch.int32),
        max_seq_len=200, causal=True, sliding_window=None, use_cascade=False,
        block_table=torch.tensor([[3, 2, 1, 0]], device="cuda", dtype=torch.int32),
    )
    out = torch.empty_like(q)
    dummy = torch.empty(qlen, 4, 256, device="cuda", dtype=torch.bfloat16)
    def run():
        impl.forward(layer, q, dummy, dummy, logical, metadata, out)
    run()
    kk = k[:200].float().repeat_interleave(6, 1) * scales[0]
    vv = v[:200].float().repeat_interleave(6, 1) * scales[1]
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk) * 0.0625
    mask = torch.arange(200, device="cuda")[None, :] <= torch.arange(qlen, device="cuda")[:, None] + 200 - qlen
    scores.masked_fill_(~mask, -torch.inf)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
    torch.testing.assert_close(out.float(), expected, atol=0.005, rtol=0.02)
    # Replay must read device-side scales instead of capturing their values.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    layer._v_scale.mul_(2)
    graph.replay()
    torch.testing.assert_close(out.float(), expected * 2, atol=0.01, rtol=0.02)
