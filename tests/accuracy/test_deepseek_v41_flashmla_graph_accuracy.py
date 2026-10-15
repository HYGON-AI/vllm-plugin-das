# SPDX-License-Identifier: Apache-2.0
"""Capture the actual adapter, including mapping/sort/mask and native planner."""

import pytest
import torch

from tests.accuracy.test_deepseek_v41_flashmla_decode_accuracy import (
    live,  # noqa: F401 -- pytest fixture
    make_cache,
    reference,
)
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import DecodeScheduler

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [None, 32, 64])
@pytest.mark.parametrize("swa_page", [32, 64])
def test_adapter_graph_replays_dynamic_inputs(live, monkeypatch, heads, page, swa_page):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH", "1")
    torch.manual_seed(20261015)
    rows = 3
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    swa, values = make_cache(swa_page)
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[0] = torch.inf
    indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(rows, 1)
    lengths = torch.full((rows,), 31, device="cuda", dtype=torch.int32)
    storage = torch.full((rows * 2, heads, 512), 7, device="cuda", dtype=q.dtype)
    args = dict(
        q=q, swa_cache=swa, kv_cache=None, swa_indices=indices,
        swa_lens=lengths, scale=512**-0.5, attn_sink=sink, output=storage[::2],
        head_dim=512, nope_head_dim=448, rope_head_dim=64,
    )
    local = table = valid = req = None
    if page is not None:
        extra, extra_values = make_cache(page)
        local = indices.clone()
        table = torch.tensor([[3, 1, 2, 0], [0, 2, 1, 3]], device="cuda", dtype=torch.int32)
        req = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        values = torch.cat((values, extra_values))
        args.update(
            kv_cache=extra, topk_indices=local, block_table=table,
            token_to_req_indices=req, is_valid_token=valid,
        )
    native, mapping, factory = runtime._decode_bindings()
    plans = []

    def record(**kwargs):
        plan = kwargs["tile_scheduler_metadata"]
        assert not plan.have_initialized, "warmup must not freeze capture's plan"
        plans.append(plan)
        return native(**kwargs)

    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (record, mapping, factory))
    args["scheduler"] = DecodeScheduler(factory, allow_graph=True)
    # Use the SAME metadata/lease for warmup and capture, exercising take()'s
    # fresh-plan path even when the runner does not rebuild between the two.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert runtime.try_flashmla_decode(**args)
        assert runtime.try_flashmla_decode(**args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        assert runtime.try_flashmla_decode(**args), "must capture native, not fallback"
    assert len(plans) == 3 and len({id(p) for p in plans}) == 3
    assert plans[-1].tile_scheduler_metadata is not None
    assert plans[-1].num_splits is not None
    captured_pointers = (plans[-1].tile_scheduler_metadata.data_ptr(), plans[-1].num_splits.data_ptr())
    columns = torch.arange(128, device="cuda")
    # Same/shrink/grow/empty/regrow, deliberately beyond captured length 31.
    for step, length in enumerate((31, 1, 128, 0, 97)):
        q.copy_(torch.randn_like(q) * 0.2)
        lengths.fill_(length)
        lengths[0] = 0  # empty/padded query row, also when other rows grow
        indices.copy_(torch.arange(128, device="cuda", dtype=torch.int32).roll(step).repeat(rows, 1))
        indices[:, -2:] = 0  # duplicates retain mass
        if local is not None:
            local.copy_(torch.arange(128, device="cuda", dtype=torch.int32).roll(step * 7).repeat(rows, 1))
            local[:, 1::3] = -1  # holes interleaved with valid entries
            local[:, -2:] = 0
            local[:, min(length, 128):] = -1
            valid[0] = False
            valid[1] = step != 3
            table.copy_(table.roll(1, -1))
            req.copy_(req.roll(1))
        original = indices.clone()
        graph.replay()
        torch.cuda.synchronize()
        expected_indices = torch.where(columns[None, :] < lengths[:, None], indices, -1)
        if local is not None:
            selected_table = table[req.long()]
            physical = selected_table.gather(1, local.clamp_min(0).long() // page) * page + local.clamp_min(0) % page
            physical = torch.where((local >= 0) & valid[:, None], physical + 4 * swa_page, -1)
            expected_indices = torch.cat((expected_indices, physical), -1)
        expected = reference(q, values, expected_indices, sink)
        torch.testing.assert_close(args["output"].float(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(indices, original)
        assert torch.all(storage[1::2] == 7)
    assert len(plans) == 3, "replay must not reenter Python or reuse warmup's plan"
    assert captured_pointers == (plans[-1].tile_scheduler_metadata.data_ptr(), plans[-1].num_splits.data_ptr())
    # A real replay must include the GPU planner, not just attention/combine.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    assert any("get_mla_metadata_kernel" in e.name for e in prof.events())
