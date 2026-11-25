# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed DeepEP receive layout and workspace sizing."""
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das.dsv41 import profile_worker
from hcu_vllm_das.dsv41.w4a8_profile import W4A8PcpProfileRunner
from hcu_vllm_das.moe import ct_w4a8_prepare as prepare
from hcu_vllm_das.moe import ct_w4a8_runtime as runtime


def config():
    return NS(
        additional_config={profile_worker.HT_FIXED_CAPACITY_KEY: True},
        parallel_config=NS(
            use_ubatching=False, prefill_context_parallel_size=8,
            pcp_shard_decode_requests=False,
        ),
        speculative_config=NS(num_speculative_tokens=5),
        scheduler_config=NS(max_num_seqs=128),
    )


@pytest.mark.parametrize("ranks,expected", [(4, 35072), (8, 38144)])
def test_capacity_uses_group_budget_and_requires_intranode(monkeypatch, ranks, expected):
    monkeypatch.setattr(profile_worker, "w4a8_profile_selected", lambda _: True)
    cfg = config()
    cfg.parallel_config.prefill_context_parallel_size = ranks
    moe = NS(max_num_tokens=32768, moe_parallel_config=NS(ep_size=ranks))
    buffer = NS(
        runtime=NS(get_num_rdma_ranks=lambda: 1),
        dispatch=lambda *, num_worst_tokens: None,
    )
    assert runtime._fixed_receive_capacity(cfg, moe, buffer) == expected
    cfg.parallel_config.pcp_shard_decode_requests = True
    assert runtime._fixed_receive_capacity(cfg, moe, buffer) == 32768
    moe.moe_parallel_config.ep_size = 1
    with pytest.raises(ValueError, match="PCP4/EP4"):
        runtime._fixed_receive_capacity(cfg, moe, buffer)
    moe.moe_parallel_config.ep_size = ranks
    buffer.runtime.get_num_rdma_ranks = lambda: 2
    with pytest.raises(ValueError, match="intranode"):
        runtime._fixed_receive_capacity(cfg, moe, buffer)
    cfg.additional_config.clear()
    assert runtime._fixed_receive_capacity(cfg, moe, buffer) is None


def test_runner_bounds_small_steps_and_profile_separately():
    runner = object.__new__(W4A8PcpProfileRunner)
    runner.vllm_config = config()
    runner.max_num_tokens, runner.max_num_reqs, runner.decode_query_len = 32768, 128, 6
    runner.pcp_manager = NS(pcp_world_size=8, shard_decode_requests=False)
    schedule = NS(total_num_scheduled_tokens=32768, num_scheduled_tokens={"request": 32768})
    assert runner._fixed_ht_capacity(schedule, False, {}) == 32810
    for tokens in (0, 1, 2, 16, 2047):
        schedule = NS(total_num_scheduled_tokens=tokens, num_scheduled_tokens={"request": tokens})
        assert runner._fixed_ht_capacity(schedule, False, {}) is None
    schedule = NS(total_num_scheduled_tokens=2048, num_scheduled_tokens={"request": 2048})
    assert runner._fixed_ht_capacity(schedule, False, {}) == 2090
    assert runner._fixed_ht_capacity(schedule, True, {"dummy_run": True}) == 38144
    assert runner._fixed_ht_capacity(schedule, False, {"dummy_run": True}) is None


def test_draft_buffer_is_separate_even_at_same_capacity_and_reused(monkeypatch):
    from vllm.distributed.device_communicators.base_device_communicator import Cache

    class Buffer:
        @staticmethod
        def get_dispatch_config(ranks):
            assert ranks == 8
            return NS(get_nvl_buffer_size_hint=lambda width, ranks: width * ranks)

        get_combine_config = get_dispatch_config

        def __init__(self, *, num_nvl_bytes, num_rdma_bytes, explicitly_destroy):
            self.num_nvl_bytes = num_nvl_bytes
            assert not num_rdma_bytes and explicitly_destroy

    monkeypatch.setitem(sys.modules, "deep_ep", NS(Buffer=Buffer))
    options = dict(num_nvl_bytes=5120 * 2 * 8, num_rdma_bytes=0, explicitly_destroy=True)
    manager = NS(world_size=8, handle_cache=Cache(), _make_all2all_kwargs=options.copy)
    target = manager.handle_cache.get_or_create(options, Buffer)
    draft = runtime._isolated_draft_ht_buffer(manager, 5120)

    assert draft is not target
    assert draft.num_nvl_bytes == target.num_nvl_bytes
    assert runtime._isolated_draft_ht_buffer(manager, 5120) is draft
    assert manager.handle_cache.get_or_create(options, Buffer) is target


@pytest.mark.parametrize("live_rows", [0, 2])
def test_fixed_dispatch_ignores_padding_and_preserves_combine_handle(monkeypatch, live_rows):
    routes = torch.full((8, 2), -1, dtype=torch.int64)
    if live_rows:
        routes[:live_rows] = torch.tensor([[0, 1], [0, -1]])
    wire = torch.full((8, 4), float("nan"), dtype=torch.bfloat16)
    wire[:live_rows] = 1
    wire_weights = torch.ones_like(routes, dtype=torch.float32)
    layout = object()
    handle = object()
    calls = []

    def get_layout(**kwargs):
        assert kwargs["async_finish"] is False and kwargs["num_experts"] == 16
        return None, None, None, None, layout

    def dispatch(**kwargs):
        assert kwargs["async_finish"] is False and kwargs["previous_event"] is layout
        assert kwargs["num_worst_tokens"] == 8 and kwargs["expert_alignment"] == 1
        calls.append("dispatch")
        event = NS(event=object(), current_stream_wait=lambda: calls.append("wait"))
        return wire, routes, wire_weights, [], handle, event

    def count(ids, local_experts, expert_map):
        assert calls[-1] == "wait" and local_experts == 2 and expert_map is None
        return torch.stack([(ids == expert).sum(dtype=torch.int32) for expert in range(local_experts)])

    def quantize(x, *args, **kwargs):
        assert torch.isfinite(x).all() and not x[live_rows:].any()
        return x.to(torch.int8), torch.ones(x.size(0), dtype=torch.float32)

    monkeypatch.setattr(prepare, "count_expert_num_tokens", count)
    monkeypatch.setattr(prepare, "moe_kernel_quantize_input", quantize)
    monkeypatch.setattr(prepare, "dbo_enabled", lambda: False)
    monkeypatch.setattr(prepare, "dbo_current_ubatch_id", lambda: 0)
    owner = prepare.HcuCTHTPrepareAndFinalize(
        NS(get_dispatch_layout=get_layout, dispatch=dispatch), 8, 1, 6,
        fixed_receive_tokens=16,
    )
    monkeypatch.setattr(owner, "_get_dispatch_config", lambda: None)
    quant = NS(quant_dtype=torch.int8, per_act_token_quant=True, block_shape=None)
    with prepare.fixed_ht_batch_capacity(8):
        receive = owner.prepare_async(
            torch.ones(1, 4, dtype=torch.bfloat16), torch.ones(1, 2),
            torch.tensor([[6, 7]]), 16, None, False, quant,
        )
    assert calls == ["dispatch"] and owner.handles[0] is handle
    q, scale, meta, received_ids, weights = receive()
    assert isinstance(meta, prepare.FixedHTExpertTokens)
    assert meta.expert_num_tokens.tolist() == ([2, 1] if live_rows else [0, 0])
    assert meta.expert_num_tokens_cpu is None
    assert received_ids is routes and weights is wire_weights
    assert q.dtype == torch.int8 and scale.shape == (8, 1)


def test_contiguous_workspace_keeps_room_for_skew_and_alignment():
    expert_counts = [257] * 51 + [38144 * 6 - 257 * 51]
    actual_rows = sum((n + 255) // 256 * 256 for n in expert_counts)
    workspace_rows = runtime.contiguous_aligned_rows(38144, 6, 52)
    assert workspace_rows % 256 == 0
    assert workspace_rows >= actual_rows


@pytest.mark.parametrize("counts, expected", [([0, 0], 0), ([1, 256, 257, 0], 1024)])
def test_lightop_device_count_matches_expert_scatter_layout(counts, expected):
    count_tensor = torch.tensor(counts, dtype=torch.int32)
    rows = runtime._aligned_expert_row_count(count_tensor, 256)
    assert rows.shape == (1,) and rows.dtype == torch.int32
    assert rows.tolist() == [expected]


def test_lightop_device_count_requires_provider_support():
    def current(x, *, num_local_tokens_tensor=None):
        return x

    def legacy(x, *, output=None):
        return x

    assert runtime._accepts_device_count(current)
    assert not runtime._accepts_device_count(legacy)
