# SPDX-License-Identifier: Apache-2.0
"""Differential checks against the full split-P/D scheduler, using real KV blocks."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.runtime_patch import test_split_pd_scheduler_v0251 as existing
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.core.sched.scheduler import HcuAsyncScheduler, HcuScheduler
from vllm_hcu.v1.core.sched.steady_decode_scheduler import (
    remember_steady_decode_state,
    supported_executor,
    try_steady_decode_schedule,
)


def make_scheduler(cls=HcuAsyncScheduler, *, max_seqs=4, policy="fcfs", blocks=128):
    def construct(**kwargs):
        config = kwargs["vllm_config"]
        config.scheduler_config.async_scheduling = issubclass(cls, HcuAsyncScheduler)
        config.scheduler_config.max_num_seqs = max_seqs
        config.scheduler_config.policy = policy
        config.model_config.max_model_len = 256
        config.model_config.model_stage = "thinker"
        config.model_config.engine_output_type = "text"
        config.model_config.stage_id = 0
        config.model_config.async_chunk = False
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.distributed_executor_backend = "mp"
        config.max_concurrent_batches = (
            2 if config.scheduler_config.async_scheduling else 1
        )
        kwargs["kv_cache_config"].num_blocks = blocks
        return cls(**kwargs)

    with patch.object(existing, "HcuScheduler", construct):
        return existing._make_scheduler()


def request(rid, *, prompt=4, output=50, priority=0):
    return Request(
        request_id=rid,
        prompt_token_ids=[1] * prompt,
        sampling_params=SamplingParams(max_tokens=output, ignore_eos=True),
        pooling_params=None,
        priority=priority,
        arrival_time=1.0,
    )


def model_output(output):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[42] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def state(scheduler):
    return {
        "current_step": scheduler.current_step,
        "running": [r.request_id for r in scheduler.running],
        "waiting": [r.request_id for r in scheduler.waiting],
        "skipped": [r.request_id for r in scheduler.skipped_waiting],
        "finished": set(scheduler.finished_req_ids),
        "preempted": set(scheduler.reset_preempted_req_ids),
        "capacity_bound": scheduler.prefill_capacity_bound,
        "free_blocks": scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        "requests": {
            rid: (
                r.status,
                r.num_computed_tokens,
                r.num_output_placeholders,
                list(r.output_token_ids),
                r.is_prefill_chunk,
                r.async_tokens_to_discard,
                scheduler.kv_cache_manager.get_block_ids(rid),
            )
            for rid, r in scheduler.requests.items()
        },
    }


@pytest.fixture(autouse=True)
def feature_env(monkeypatch):
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_PD_SPLIT", True, raising=False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True, raising=False)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_STEADY_DECODE_SCHED_FASTPATH", False, raising=False
    )


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("backlog", [False, True])
@pytest.mark.parametrize("max_output", [1, 2, 20, 50])
def test_fast_and_full_two_inflight_batches_match(
    monkeypatch, policy, backlog, max_output
):
    full = make_scheduler(policy=policy)
    fast = make_scheduler(policy=policy)
    count = 8 if backlog else 4
    for scheduler in (full, fast):
        for i in range(count):
            scheduler.add_request(request(str(i), output=max_output, priority=i % 2))
    queues = [deque(), deque()]
    for step in range(400):
        outputs = []
        for scheduler, queue, enable in zip((full, fast), queues, (False, True)):
            monkeypatch.setattr(henvs, "VLLM_HCU_STEADY_DECODE_SCHED_FASTPATH", enable)
            output = scheduler.schedule()
            outputs.append(asdict(output))
            queue.append(output)
        assert outputs[0] == outputs[1], step
        assert state(full) == state(fast), step
        # Reproduce native queue=2: submit the next batch before processing
        # the oldest output. Empty batches still participate in lifecycle work.
        if len(queues[0]) >= 2:
            for scheduler, queue in zip((full, fast), queues):
                output = queue.popleft()
                scheduler.update_from_output(output, model_output(output))
            assert state(full) == state(fast), step
        if not full.requests and not fast.requests:
            break
    else:
        pytest.fail("requests did not finish")
    if max_output >= 20:
        assert fast._vllm_hcu_steady_decode_hits > 0
    assert full.kv_cache_manager.block_pool.get_num_free_blocks() == 127


def primed():
    scheduler = make_scheduler()
    for i in range(4):
        scheduler.add_request(request(str(i)))
    for _ in range(2):
        out = scheduler.schedule()
        scheduler.update_from_output(out, model_output(out))
    remember_steady_decode_state(scheduler, out)
    assert scheduler._vllm_hcu_steady_decode_state is not None
    return scheduler


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("reorder", "request_state_changed"),
        ("discard", "request_state_changed"),
        ("finished", "request_lifecycle"),
        ("preempted", "request_lifecycle"),
        ("streaming", "streaming_input"),
        ("page", "allocation_required"),
        ("context", "request_state_changed"),
        ("encoder", "request_state_changed"),
    ],
)
def test_rejection_has_no_scheduling_or_kv_side_effect(mutation, reason):
    scheduler = primed()
    r = scheduler.running[0]
    if mutation == "reorder":
        scheduler.running.reverse()
    elif mutation == "discard":
        r.async_tokens_to_discard = 1
    elif mutation == "finished":
        scheduler.finished_req_ids.add("finished")
    elif mutation == "preempted":
        scheduler.reset_preempted_req_ids.add("preempted")
    elif mutation == "streaming":
        scheduler.num_waiting_for_streaming_input = 1
    elif mutation == "page":
        manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        manager.req_to_blocks[r.request_id] = []
    elif mutation == "context":
        r.num_computed_tokens = scheduler.max_model_len - 1
    elif mutation == "encoder":
        r.mm_features = [
            SimpleNamespace(mm_position=SimpleNamespace(offset=0, length=100))
        ]
    before = state(scheduler)
    assert try_steady_decode_schedule(scheduler) is None
    assert scheduler._vllm_hcu_steady_decode_last_reason == reason
    assert state(scheduler) == before


def test_noop_fastpath_does_not_call_allocator(monkeypatch):
    scheduler = primed()

    def forbidden(*args, **kwargs):
        raise AssertionError("allocator called")

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", forbidden)
    before = scheduler.current_step
    result = try_steady_decode_schedule(scheduler)
    assert result.total_num_scheduled_tokens == 4
    assert scheduler.current_step == before + 1


def test_new_admission_invalidates_fastpath():
    scheduler = primed()
    scheduler.max_num_running_reqs = 5
    scheduler.add_request(request("new"))
    before = state(scheduler)
    assert try_steady_decode_schedule(scheduler) is None
    assert scheduler._vllm_hcu_steady_decode_last_reason == "waiting_admission"
    assert state(scheduler) == before


def test_async_mro_keeps_native_placeholder_hooks():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    assert (
        HcuAsyncScheduler._update_after_schedule
        is AsyncScheduler._update_after_schedule
    )
    assert (
        HcuAsyncScheduler._update_request_with_output
        is AsyncScheduler._update_request_with_output
    )
    scheduler = make_scheduler()
    scheduler.add_request(request("a"))
    output = scheduler.schedule()
    assert scheduler.requests["a"].num_output_placeholders == 1
    scheduler.update_from_output(output, model_output(output))
    assert scheduler.requests["a"].num_output_placeholders == 0


@pytest.mark.parametrize(
    "path",
    [
        "mp",
        "uni",
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
        "vllm.v1.executor.uniproc_executor.UniProcExecutor",
        "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor",
    ],
)
def test_executor_selection_accepts_native_and_hcu_lazy_paths(path):
    assert supported_executor(path)
    assert not supported_executor("unvalidated.custom.Executor")


def test_fastpath_rejects_disabled_split_pd(monkeypatch):
    monkeypatch.setattr(henvs, 'VLLM_HCU_USE_PD_SPLIT', False)
    monkeypatch.setattr(henvs, 'VLLM_HCU_STEADY_DECODE_SCHED_FASTPATH', True)
    with pytest.raises(ValueError, match='requires VLLM_HCU_USE_PD_SPLIT=1'):
        make_scheduler()
