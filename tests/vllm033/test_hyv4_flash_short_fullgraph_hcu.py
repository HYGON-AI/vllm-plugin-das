# SPDX-License-Identifier: Apache-2.0
"""Actual HIP selector/unpruned replay equivalence; not whole-model proof.

No attention/cache/provider doubles. Use exactly one assigned HCU device.
Missing hardware, activation or kernels fail rather than skip.
"""
from importlib import import_module, metadata
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.hcu


@pytest.fixture(scope="session")
def ops():
    assert torch.version.hip is not None and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin
    hcu_runtime_plugin()
    module = import_module("hcu_vllm_das.models.hy_v4_flash.indexpool_ops")
    root = Path(metadata.distribution("hcu-vllm-das-plugin").locate_file("")).resolve()
    assert Path(module.__file__).resolve().is_relative_to(root)
    assert module.is_hcu_platform()
    return module


@pytest.mark.parametrize("ratio,budget", [
    (2, 511), (4, 511), (8, 511), (16, 511),
    pytest.param(128, 31, id="operator-only-ratio128"),
])
def test_short_causal_indices_equal_full_selector_after_changed_input_replay(ops, ratio, budget):
    bound = (budget + 1) * ratio
    positions = torch.empty(8, device="cuda", dtype=torch.int64)
    active = torch.empty(8, device="cuda", dtype=torch.bool)
    # Same conservative scoring width used by the original million-token path.
    width = ((1048576 + ratio * 64 - 1) // (ratio * 64)) * 64
    torch.manual_seed(ratio)
    scores = torch.randn(8, width, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def wide_path():
        groups = ops.select_groups(scores, positions // ratio, budget)
        return ops.expand_groups(groups, positions, ratio, active)

    def short_path():
        return ops.unpruned_indices(positions, bound, active)

    with torch.cuda.stream(stream):
        positions.fill_(0)
        active.fill_(True)
        for _ in range(3):
            wide_path()
            short_path()
    stream.synchronize()
    wide_graph, short_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    from vllm.config import CUDAGraphMode
    from hcu_vllm_das.models.hy_v4_flash.fullgraph_banks import FlashFullGraphBanks
    # Exercise production pool ownership, including Core's profiling pool swap.
    shared = torch.cuda.graph_pool_handle()
    wide_manager = NS(pool=shared, _capture_descs={CUDAGraphMode.FULL: [8]})
    short_manager = NS(pool=shared, _capture_descs={CUDAGraphMode.FULL: [8]})
    owner = NS(group_budget=budget, ratio=ratio, max_model_len=1048576)
    banks = FlashFullGraphBanks(
        wide_manager, short_manager, [owner], bound, CUDAGraphMode.FULL,
        short_pool_factory=torch.cuda.graph_pool_handle,
    )
    banks.pool = torch.cuda.graph_pool_handle()
    assert wide_manager.pool != short_manager.pool
    with torch.cuda.graph(wide_graph, pool=wide_manager.pool, stream=stream):
        wide = wide_path()
    with torch.cuda.graph(short_graph, pool=short_manager.pool, stream=stream):
        short = short_path()
    addresses = wide.data_ptr(), short.data_ptr()
    for values, flags in [
        ([-1, 0, 1, ratio - 1, ratio, bound - 2, bound - 1, 31], [True] * 8),
        ([bound - 1, ratio, 0, -1, 2, bound - 2, 1, 15], [True, False] * 4),
        ([0, bound - 1, 31, ratio + 1, -1, 7, 11, 1], [False, True] * 4),
    ]:
        with torch.cuda.stream(stream):
            positions.copy_(torch.tensor(values, device="cuda"))
            active.copy_(torch.tensor(flags, device="cuda"))
            scores.normal_()
            wide_graph.replay()
            short_graph.replay()
        stream.synchronize()
        assert (wide.data_ptr(), short.data_ptr()) == addresses
        expected = torch.tensor([[j if enabled and 0 <= j <= pos else -1
                                  for j in range(bound)]
                                 for pos, enabled in zip(values, flags)], dtype=torch.int32)
        torch.testing.assert_close(wide.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(short.cpu(), expected, atol=0, rtol=0)
    # A retained output from one bank must survive a replay of the other.
    # Shared graph pools must not reclaim live result storage across banks.
    retained_wide = wide.cpu().clone()
    with torch.cuda.stream(stream):
        positions.fill_(bound - 1)
        active.fill_(True)
        short_graph.replay()
    stream.synchronize()
    torch.testing.assert_close(wide.cpu(), retained_wide, atol=0, rtol=0)
    expected_short = torch.arange(bound, dtype=torch.int32).repeat(8, 1)
    torch.testing.assert_close(short.cpu(), expected_short, atol=0, rtol=0)
    retained_short = short.cpu().clone()
    with torch.cuda.stream(stream):
        active.fill_(False)
        wide_graph.replay()
    stream.synchronize()
    assert (wide == -1).all().item()
    torch.testing.assert_close(short.cpu(), retained_short, atol=0, rtol=0)


@pytest.mark.parametrize("ratio,budget", [
    (2, 511), (4, 511), (8, 511), (16, 511),
    pytest.param(128, 31, id="operator-only-ratio128"),
])
def test_above_bound_requires_wide_path_to_keep_current_tail(ops, ratio, budget):
    bound = (budget + 1) * ratio
    positions = torch.tensor([bound], device="cuda")
    scores = torch.arange(budget + 2, device="cuda", dtype=torch.float32)[None]
    groups = ops.select_groups(scores, positions // ratio, budget)
    wide = ops.expand_groups(groups, positions, ratio)
    short = ops.unpruned_indices(positions, bound)
    assert (wide == bound).any().item()
    assert not (short == bound).any().item()
    assert not torch.equal(wide, short)
