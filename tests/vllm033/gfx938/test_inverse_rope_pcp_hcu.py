# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicitly device-bound numerical/changed-input Graph operator checks."""
from statistics import median

import pytest
import torch

def _time_events(function, prepare):
    for _ in range(10):
        prepare()
        function()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(50):
        prepare()
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


@pytest.mark.hcu
@pytest.mark.parametrize("rows,heads", [(1, 1), (8, 2), (128, 64), (4096, 64)])
@torch.inference_mode()
def test_inverse_rope_matches_core_eager_and_changed_input_graph(rows, heads, record_property):
    from hcu_vllm_das.inverse_rope_hcu import prepare_operator
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_rows_

    assert torch.version.hip is not None and torch.cuda.is_available(), "Expected a HIP device"
    torch.cuda.set_device(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx938", f"Inverse RoPE numerical contract requires gfx938, got {arch}"

    operator = prepare_operator(rocm_inverse_rope_rows_)
    record_property("native_fuse_cosine", operator.keywords["fuse_cosine"])
    output = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    before = output.clone()
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    angles = torch.randn(rows + 1, 32, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    expected = before.clone()
    rocm_inverse_rope_rows_(expected, positions, cache, 64)
    operator(output[..., -64:], positions, cache)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(output[..., :448], before[..., :448], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operator(output[..., -64:], positions, cache)
    for seed in (42, 43):
        torch.manual_seed(seed)
        output.normal_()
        positions.copy_((positions + 1) % cache.shape[0])
        cache.neg_()
        expected.copy_(output)
        rocm_inverse_rope_rows_(expected, positions, cache, 64)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    source = output.clone()

    def prepare():
        output.copy_(source)
        expected.copy_(source)

    stock_ms = _time_events(lambda: rocm_inverse_rope_rows_(expected, positions, cache, 64), prepare)
    native_ms = _time_events(lambda: operator(output[..., -64:], positions, cache), prepare)
    record_property("stock_p50_ms", stock_ms)
    record_property("hcu_p50_ms", native_ms)
    record_property("stock_over_hcu", stock_ms / native_ms)


@pytest.mark.hcu
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_inverse_rope_cancellation_matches_core_and_fp64_oracle(position_dtype, record_property):
    """Frozen rounding witnesses must survive eager and changed-input Graph."""
    from hcu_vllm_das.inverse_rope_hcu import prepare_operator
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_rows_

    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx938"
    # Actual BF16 lane pairs and FP32 coefficients from a failing replay.
    values = torch.tensor([
        [-1.1484375, 0.83203125, -0.809720516204834, 0.5868157148361206],
        [0.546875, -2.359375, -0.22580166161060333, 0.9741733074188232],
        [-0.80078125, -1.0390625, -0.7926467657089233, 0.6096811890602112],
    ], dtype=torch.float32)
    rows = len(values)
    source = torch.zeros(rows, 1, 512, dtype=torch.bfloat16)
    source[..., :448] = torch.arange(448, dtype=torch.bfloat16)
    source[..., 448::2] = values[:, 0, None, None]
    source[..., 449::2] = values[:, 1, None, None]
    initial_cache = torch.cat((values[:, 2, None].expand(-1, 32),
                               values[:, 3, None].expand(-1, 32)), dim=-1)
    output = source.cuda()
    positions = torch.arange(rows, device="cuda", dtype=position_dtype)
    cache = initial_cache.cuda()
    operator = prepare_operator(rocm_inverse_rope_rows_)
    record_property("native_fuse_cosine", operator.keywords["fuse_cosine"])

    def reference(cpu_source, cpu_positions, cpu_cache, fuse_cosine):
        a = cpu_source[..., 448::2].float()
        b = cpu_source[..., 449::2].float()
        c = cpu_cache[cpu_positions.long(), :32, None].transpose(1, 2)
        s = cpu_cache[cpu_positions.long(), 32:, None].transpose(1, 2)
        # Represent each product exactly in FP64. Both independently derived
        # contracts remain exact; the actual Core result selects one below.
        even_cos, odd_cos = a.double() * c.double(), b.double() * c.double()
        if not fuse_cosine:
            even_cos, odd_cos = even_cos.float().double(), odd_cos.float().double()
        even = (even_cos + (b * s).double()).float()
        odd = (odd_cos - (a * s).double()).float()
        expected = cpu_source.clone()
        expected[..., 448::2] = even.to(torch.bfloat16)
        expected[..., 449::2] = odd.to(torch.bfloat16)
        return expected.cuda()

    def check(cpu_source, cpu_positions, cpu_cache, graph=None):
        output.copy_(cpu_source)
        positions.copy_(cpu_positions)
        cache.copy_(cpu_cache)
        stock = output.clone()
        rocm_inverse_rope_rows_(stock, positions, cache, 64)
        candidates = [reference(cpu_source, cpu_positions, cpu_cache, mode)
                      for mode in (False, True)]
        matching = [oracle for oracle in candidates if torch.equal(stock, oracle)]
        assert len(matching) == 1, "Core must match exactly one independent rounding contract"
        oracle = matching[0]
        torch.testing.assert_close(stock, oracle, rtol=0, atol=0)
        if graph is None:
            operator(output[..., -64:], positions, cache)
        else:
            graph.replay()
        torch.testing.assert_close(output, oracle, rtol=0, atol=0)
        torch.testing.assert_close(output[..., :448], stock[..., :448], rtol=0, atol=0)

    cpu_positions = torch.arange(rows, dtype=position_dtype)
    check(source, cpu_positions, initial_cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operator(output[..., -64:], positions, cache)
    for wave in (1, 2):
        changed = source.roll(wave, dims=0)
        changed[..., 448:] *= -1 if wave == 1 else 2
        check(changed, cpu_positions.roll(wave), -initial_cache, graph)


@pytest.mark.hcu
@pytest.mark.parametrize("fuse_cosine", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_raw_native_modes_match_independent_oracles_and_graph(fuse_cosine, position_dtype):
    from hcu_vllm_das.inverse_rope_hcu import load_operator

    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx938"
    operator = load_operator()
    source = torch.tensor([-1.1484375, 0.83203125], dtype=torch.bfloat16).repeat(32).reshape(1, 1, 64)
    cache_cpu = torch.tensor([-0.809720516204834] * 32 + [0.5868157148361206] * 32).reshape(1, 64)
    output, cache = source.cuda(), cache_cpu.cuda()
    positions = torch.zeros(1, device="cuda", dtype=position_dtype)
    operator(output, positions, cache, fuse_cosine=fuse_cosine)
    output.copy_(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operator(output, positions, cache, fuse_cosine=fuse_cosine)
    for wave in (1, -1, 2):
        changed = source * wave
        a, b = changed[..., ::2].float(), changed[..., 1::2].float()
        c, s = cache_cpu[:, :32].reshape(1, 1, 32), cache_cpu[:, 32:].reshape(1, 1, 32)
        ac, bc = a.double() * c.double(), b.double() * c.double()
        if not fuse_cosine:
            ac, bc = ac.float().double(), bc.float().double()
        expected = changed.clone()
        expected[..., ::2] = (ac + (b * s).double()).float().to(torch.bfloat16)
        expected[..., 1::2] = (bc - (a * s).double()).float().to(torch.bfloat16)
        output.copy_(changed)
        operator(output, positions, cache, fuse_cosine=fuse_cosine)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        output.copy_(changed)
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
