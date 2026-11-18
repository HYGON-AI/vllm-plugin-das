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
    from hcu_vllm_das.inverse_rope_hcu import load_operator
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_rows_

    assert torch.version.hip is not None and torch.cuda.is_available(), "Expected a HIP device"
    torch.cuda.set_device(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx938", f"Inverse RoPE numerical contract requires gfx938, got {arch}"

    operator = load_operator()
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
