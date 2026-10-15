# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Live gfx93x numerical tests of the V4.1 FlashMLA dispatch and graph replay."""

from __future__ import annotations

import pytest
import torch

from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla as runtime

pytestmark = pytest.mark.hcu


def sparse_reference(q, kv, indices, lengths, sink, scale):
    """FP32 attention with shared K/V, sink mass, and explicit invalid masks."""
    indices = indices.reshape(q.shape[0], -1).long()
    valid = (indices >= 0) & (indices < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(indices.shape[-1], device=q.device)[None, :] < lengths[:, None]
    selected = kv[:, 0][indices.clamp(0, kv.shape[0] - 1)].float()
    # Mask before dot-products so NaNs in unused gathered workspace cannot leak.
    selected = torch.where(valid[..., None], selected, 0)
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    if sink is not None:
        sink = sink[:q.shape[1]]
        scores = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
        selected = torch.cat((selected, torch.zeros_like(selected[:, :1])), 1)
    weights = torch.softmax(scores, -1).nan_to_num()
    result = torch.einsum("thk,tkd->thd", weights, selected)
    if sink is not None:
        result[:, torch.isposinf(sink)] = 0
    return result


def _require_gfx93x():
    if not torch.cuda.is_available() or not runtime._supported_device(torch.empty(0, device="cuda")):
        pytest.skip("native FlashMLA prefill test requires validated gfx936/gfx938")
    if runtime._flashmla_prefill() is None:
        pytest.skip("native FlashMLA sparse library unavailable")


@pytest.mark.parametrize("tokens,heads,width", [
    (1, 4, 128), (33, 8, 256), (129, 16, 384), (16, 64, 128), (2, 128, 128),
])
@pytest.mark.parametrize("with_sink", [False, True])
def test_native_prefill_matches_fp32_reference(monkeypatch, tokens, heads, width, with_sink):
    _require_gfx93x()
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_PREFILL", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    generator = torch.Generator(device="cuda").manual_seed(20261013 + tokens)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator)
    kv = torch.randn(1024, 1, 512, device="cuda", dtype=torch.bfloat16, generator=generator)
    indices = torch.randint(0, 1023, (tokens, width), device="cuda", dtype=torch.int32, generator=generator)
    lengths = torch.arange(tokens, device="cuda", dtype=torch.int32) % (width - 1) + 1
    # Mixed request offsets, invalid sentinels, and ignored-but-valid suffixes.
    indices[:, :1] = -1
    if tokens > 1:
        lengths[0] = 0
        indices[1, :2] = 1024
    kv[-1] = float("nan")
    indices[:, -1] = 1023
    sink = torch.linspace(-3, 3, heads + 1, device="cuda") if with_sink else None
    if sink is not None:
        sink[0] = -torch.inf
        sink[-2] = torch.inf
    indices_before = indices.clone()
    # Narrowed output rows must not overwrite the decode section of the buffer.
    storage = torch.full((tokens + 2, heads, 512), 7, device="cuda", dtype=torch.bfloat16)
    output = storage[1:-1]
    assert runtime.try_flashmla_prefill(
        q, kv, indices, lengths, 512**-0.5, 512, 448, 64, sink, output,
    )
    torch.cuda.synchronize()
    expected = sparse_reference(q, kv, indices, lengths, sink, 512**-0.5)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.02)
    torch.testing.assert_close(indices, indices_before)
    assert torch.all(storage[0] == 7) and torch.all(storage[-1] == 7)


def test_native_prefill_graph_replay_with_strided_inputs(monkeypatch):
    _require_gfx93x()
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_PREFILL", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    q = torch.randn(66, 16, 512, device="cuda", dtype=torch.bfloat16)[::2]
    kv = torch.randn(1024, 1, 512, device="cuda", dtype=torch.bfloat16)[::2]
    indices = torch.randint(0, 512, (33, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((66,), 128, device="cuda", dtype=torch.int32)[::2]
    sink = torch.zeros(16, device="cuda")
    output = torch.empty_like(q)
    def run():
        assert runtime.try_flashmla_prefill(
            q, kv, indices, lengths, 512**-0.5, 512, 448, 64, sink, output,
        )
    # Warm lazy resolution, extension initialization, and logging before capture.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.5)
    lengths.fill_(31)
    graph.replay()
    torch.cuda.synchronize()
    expected = sparse_reference(q, kv, indices, lengths, sink, 512**-0.5)
    torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.02)
