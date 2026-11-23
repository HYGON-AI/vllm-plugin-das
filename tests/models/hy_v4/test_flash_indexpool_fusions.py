# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact IndexPool fusion math, compaction and changing-input Graph probes."""
import pytest
import torch

from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_ops import (
    _expand_groups_torch, _fwht_quantize_torch, expand_groups, fwht_quantize,
)

HCU = torch.cuda.is_available() and torch.version.hip is not None
requires_hcu = pytest.mark.skipif(not HCU, reason="requires HCU Triton")


@pytest.mark.parametrize("shape", [(0, 128), (2, 128), (32, 128)])
def test_cpu_dispatch_retains_reference(shape):
    x = torch.randn(shape).bfloat16()
    actual = fwht_quantize(x)
    expected = _fwht_quantize_torch(x)
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@requires_hcu
@pytest.mark.parametrize("rows", [0, 1, 2, 32, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_fwht_exact_and_input_unchanged(rows, strided):
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_triton import fwht_quantize_triton
    torch.manual_seed(rows)
    x = torch.randn(rows, 256 if strided else 128, device="cuda").bfloat16()
    if strided:
        x = x[:, ::2]
    before = x.clone()
    expected = _fwht_quantize_torch(x)
    actual = fwht_quantize_triton(x)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(x, before)
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[1].dtype == torch.float32


@requires_hcu
def test_fwht_zero_tiny_large_and_scale_boundaries():
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_triton import fwht_quantize_triton
    # BF16 neighborhoods on both sides of power-of-two scale transitions.
    x = torch.zeros(96, 128, device="cuda", dtype=torch.bfloat16)
    for k in range(30):
        x[k * 3:k * 3 + 3, 0] = torch.tensor([0.9921875, 1.0, 1.0078125], device="cuda") * (448 * 128**0.5 * 2.0**(k-20))
    x[-5].fill_(1e-20)
    x[-4].fill_(-1e12)
    x[-3].fill_(float("nan"))
    x[-2, 0] = float("inf")
    x[-1, 0] = -float("inf")
    a, s = fwht_quantize_triton(x)
    b, t = _fwht_quantize_torch(x)
    torch.testing.assert_close(s, t, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0, equal_nan=True)


@requires_hcu
@pytest.mark.parametrize("ratio,budget", [(1, 0), (4, 7), (4, 511), (3, 17)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("use_active", [False, True])
def test_expansion_exact_at_long_context_boundaries(ratio, budget, strided, use_active):
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_triton import expand_groups_triton
    positions = torch.tensor([-1, 0, 31, 2047, 2048, 4095, 4096, 8191], device="cuda")
    groups = torch.arange(budget, device="cuda")[None].expand(8, -1).clone().int()
    if budget:
        groups[:, ::5] = -1
        groups[:, ::7] = 50000
    if strided:
        buf = torch.full((8, budget * 2), 123, device="cuda", dtype=torch.int32)
        buf[:, ::2] = groups
        groups = buf[:, ::2]
        bufp = torch.zeros(16, device="cuda", dtype=torch.long)
        bufp[::2] = positions
        positions = bufp[::2]
    active = torch.tensor([True, True, False, True, True, False, True, True], device="cuda") if use_active else None
    expected = _expand_groups_torch(groups, positions, ratio, active)
    actual = expand_groups_triton(groups, positions, ratio, active)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(expand_groups(groups, positions, ratio, active), expected, rtol=0, atol=0)


@requires_hcu
def test_fusions_changing_input_graph_and_output_ownership():
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_triton import fwht_quantize_triton, expand_groups_triton
    x = torch.randn(32, 128, device="cuda").bfloat16()
    groups = torch.arange(511, device="cuda", dtype=torch.int32)[None].repeat(2, 1)
    positions = torch.tensor([4095, 8191], device="cuda")
    active = torch.ones(2, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fwht_quantize_triton(x)
            expand_groups_triton(groups, positions, 4, active)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        quant, scales = fwht_quantize_triton(x)
        indices = expand_groups_triton(groups, positions, 4, active)
    for step in range(3):
        x.copy_(torch.randn_like(x) * 2**step)
        groups[:, ::3] = -step - 1
        positions.add_(1)
        active[1] = step != 1
        graph.replay()
        q, s = _fwht_quantize_torch(x)
        torch.testing.assert_close(scales, s, rtol=0, atol=0)
        assert torch.equal(quant.view(torch.uint8), q.view(torch.uint8))
        torch.testing.assert_close(indices, _expand_groups_torch(groups, positions, 4, active), rtol=0, atol=0)
    old = quant.clone().view(torch.uint8).clone()
    fwht_quantize_triton(x * 3)
    assert torch.equal(quant.view(torch.uint8), old)


@requires_hcu
def test_expansion_broadcast_and_empty_dispatch():
    groups = torch.arange(7, device="cuda", dtype=torch.int32)[None]
    positions = torch.tensor([31, 4095], device="cuda")
    # Torch's broadcasting face is intentionally retained, not guessed native.
    torch.testing.assert_close(expand_groups(groups, positions, 4),
                               _expand_groups_torch(groups, positions, 4), rtol=0, atol=0)
    groups = torch.empty(0, 7, device="cuda", dtype=torch.int32)
    positions = torch.empty(0, device="cuda", dtype=torch.long)
    assert expand_groups(groups, positions, 4).shape == (0, 32)


@requires_hcu
def test_long_pool_writer_and_native_scorer_8192_bound():
    from vllm.v1.worker.workspace import init_workspace_manager
    from hcu_adapter.bridges.vllm_paged_mqa import get_paged_mqa_provider
    from hcu_adapter.kernels.mla_attention_emu import fp8_fp4_paged_mqa_logits
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_ops import write_pool

    init_workspace_manager(torch.device("cuda:0"))
    torch.manual_seed(8192)
    width, heads = 2048, 32  # ratio4, max_model_len8192
    pages = width // 64
    backing = torch.full((2 * pages, 64 * 132 + 256), 67,
                         device="cuda", dtype=torch.uint8)
    cache = backing[:, 128:128 + 64 * 132].view(2 * pages, 64, 132)
    table = torch.randperm(2 * pages, device="cuda").int().reshape(2, pages)
    owners = torch.arange(2, device="cuda").repeat_interleave(width)
    groups = torch.arange(width, device="cuda").repeat(2)
    active = torch.ones(2 * width, device="cuda", dtype=torch.bool)
    quant, scales = fwht_quantize(torch.randn(2 * width, 128, device="cuda").bfloat16())
    write_pool(cache, table, owners, groups, active, quant, scales)
    q, qs = fwht_quantize(torch.randn(2 * heads, 128, device="cuda").bfloat16())
    q = q.reshape(2, 1, heads, 128)
    weights = torch.randn(2, heads, device="cuda") * qs.reshape(2, heads)
    lengths = torch.tensor([[2047], [1024]], device="cuda", dtype=torch.int32)
    metadata, score = get_paged_mqa_provider()
    actual = score((q, None), cache.unsqueeze(2), weights, lengths, table,
                   metadata(lengths, 64, 2), max_model_len=width)
    expected = fp8_fp4_paged_mqa_logits((q, None), cache.unsqueeze(2), weights,
                                       lengths, table, None, width)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert torch.count_nonzero(actual[0, 2047:]) == 0
    assert torch.count_nonzero(actual[1, 1024:]) == 0
    assert torch.all(backing[:, :128] == 67)
    assert torch.all(backing[:, 128 + 64 * 132:] == 67)


@requires_hcu
def test_expansion_preserves_reference_int32_cast_order():
    groups = torch.tensor([[0, 2**29, 2**30, -1]], device="cuda")
    positions = torch.tensor([2**33], device="cuda")
    torch.testing.assert_close(expand_groups(groups, positions, 4),
                               _expand_groups_torch(groups, positions, 4), rtol=0, atol=0)
