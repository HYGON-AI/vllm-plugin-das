# SPDX-License-Identifier: Apache-2.0
"""Exercise the upstream scheduler through PCP isolation and worker partitioning."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.request import Request, RequestStatus
from vllm_hcu.patch.platform.framework_opt import patch_scheduler as patch
from tests.runtime_patch.test_glm52_pcp_manager import (
    _InMemoryBlockTables, _make_batch, _make_managers,
)


def _request(name, prompt=64, total=None, computed=0, priority=0):
    request = Request(name, list(range(prompt)), SamplingParams(max_tokens=8192), None,
                      priority=priority)
    if total is not None:
        request.append_output_token_ids([7] * (total - prompt))
    request.num_computed_tokens = computed
    return request


def _scheduler(running, waiting=(), *, spec_tokens=0, policy=SchedulingPolicy.FCFS,
               isolate=True):
    # Run the actual upstream schedule(), replacing KV allocation only.
    class TestScheduler(Scheduler):
        pass
    patch._install_pcp_prefill_isolation(TestScheduler)
    scheduler = TestScheduler.__new__(TestScheduler)
    scheduler.running = list(running)
    scheduler.waiting = create_request_queue(policy)
    scheduler.skipped_waiting = create_request_queue(policy)
    scheduler.policy = policy
    for request in waiting:
        scheduler._enqueue_waiting_request(request)
    for request in running:
        request.status = RequestStatus.RUNNING
    scheduler.requests = {r.request_id: r for r in [*running, *waiting]}
    scheduler.current_step = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 2048
    scheduler.max_model_len = 16384
    scheduler.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=2048, long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
    )
    scheduler.vllm_config = SimpleNamespace(speculative_config=None)
    scheduler.num_spec_tokens = spec_tokens
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.num_prefill_lookahead = 1
    scheduler.num_lookahead_tokens = spec_tokens
    scheduler.dynamic_sd_lookup = None
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.prefill_capacity_bound = False
    scheduler.lora_config = None
    scheduler.ec_connector = scheduler.connector = None
    scheduler.need_mamba_block_aligned_split = False
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.log_stats = False
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = scheduler.model_uses_xdrope = False
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler._inflight_prefills = set()
    scheduler.grammar_compile_error_reqs = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.sched_step_seq = 0
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler.kv_cache_manager = Mock()
    blocks = Mock()
    blocks.get_block_ids.return_value = ([1],)
    scheduler.kv_cache_manager.empty_kv_cache_blocks = blocks
    scheduler.kv_cache_manager.allocate_slots.return_value = blocks
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (blocks, 0, 0)
    scheduler.kv_cache_manager.get_num_common_prefix_blocks.return_value = [0]
    scheduler.kv_cache_manager.take_boundary_state_offloads.return_value = {}
    scheduler.kv_cache_manager.take_kv_cache_block_copies.return_value = ([], [])
    scheduler.encoder_cache_manager = Mock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.encoder_cache_manager.get_manager_metadata.return_value = None
    scheduler._get_new_block_ids_to_zero = lambda: None
    scheduler._ec_transfer_pending = lambda *args: False
    scheduler._make_cached_request_data = lambda *args: CachedRequestData.make_empty()
    setattr(scheduler, patch._ISOLATE_PREFILL, isolate)
    return scheduler


@pytest.mark.parametrize("spec_tokens", [1, 5, 7])
@pytest.mark.parametrize("pcp_size", [2, 4, 8])
def test_scheduler_single_token_prompt_partitions_on_all_pcp_ranks(spec_tokens, pcp_size):
    decode = _request("decode", total=80, computed=79)
    width = spec_tokens + 1
    decode.spec_token_ids = [3] * spec_tokens
    new = _request("new", prompt=1)
    scheduler = _scheduler([decode], [new], spec_tokens=spec_tokens)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"decode": width, "new": width}
    assert output.scheduled_spec_decode_tokens["new"] == [-1] * spec_tokens
    # Use the upstream MRV2 gather to retain its real phase classification.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    runner = SimpleNamespace(
        decode_query_len=width, adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"decode": 0, "new": 1},
            prefill_len=SimpleNamespace(np=np.array([64, 1], dtype=np.int32)),
            num_computed_prefill_tokens=np.array([64, 0], dtype=np.int32),
        ),
    )
    gathered, uniform = GPUModelRunner.gather_batch_req_state(runner, output, False)
    assert gathered.req_ids == ["decode", "new"]
    assert gathered.is_prefilling_np.tolist() == [False, True]
    assert uniform is None
    from tests.runtime_patch.test_pcp_pre_dispatch import dispatch_before_prepare
    managers, groups = _make_managers(pcp_size, block_tables=_InMemoryBlockTables())
    runner.pcp_manager = managers[0]
    dispatch_padding, actual_state, actual_uniform = dispatch_before_prepare(runner, output)
    assert dispatch_padding == 2 * width
    assert actual_state.is_prefilling_np.tolist() == [False, True]
    assert actual_uniform is None
    # The execute scope must not leak metadata to later two-array calls.
    assert managers[0].get_num_tokens_for_dispatch(
        actual_state.num_scheduled_tokens, actual_state.is_prefilling_np
    ) < dispatch_padding
    requests = []
    offset = 0
    for index, req_id in enumerate(gathered.req_ids):
        width = int(gathered.num_scheduled_tokens[index])
        tokens = list(range(offset, offset + width))
        seq_len = 79 + width if req_id == "decode" else width
        requests.append((req_id, tokens, seq_len, bool(gathered.is_prefilling_np[index])))
        offset += width
    batch = _make_batch(requests)
    batch.prefill_len_np = gathered.prefill_len_np
    batch.num_computed_prefill_tokens_np = gathered.num_computed_prefill_tokens_np
    batch.num_draft_tokens_per_req = np.array([
        len(output.scheduled_spec_decode_tokens.get(req_id, ()))
        for req_id in gathered.req_ids
    ], dtype=np.int32)
    batch.num_draft_tokens = int(batch.num_draft_tokens_per_req.sum())
    batch.cu_num_logits_np = np.concatenate(([0], np.cumsum(batch.num_draft_tokens_per_req + 1))).astype(np.int32)
    batch.cu_num_logits = torch.from_numpy(batch.cu_num_logits_np.copy())
    batch.logits_indices = torch.arange(int(batch.cu_num_logits_np[-1]))
    assert managers[0].get_num_tokens_for_dispatch(batch) == dispatch_padding
    assert dispatch_padding == 2 * width
    local = [
        m.partition_batch(batch, padded_num_tokens=dispatch_padding)
        for m in managers
    ]
    for manager, item in zip(managers, local):
        assert item.num_scheduled_tokens.tolist() == [width, width]
        assert item.cu_num_logits_np.tolist() == [0, width, 2 * width]
        assert item.num_draft_tokens_per_req.tolist() == [spec_tokens, spec_tokens]
        assert not item.is_prefilling_np.any()
        assert not item.is_padding.any()
        manager.prepare_attn(item)
        assert item._vllm_hcu_pcp_replicated_token_mask.all()
        # Every replicated token has one canonical writer, on rank 0 only.
        assert int(manager._gathered_kv_write_mask.sum()) == batch.num_tokens
    # The original MRV2 evidence is retained for global backend/graph dispatch.
    assert batch.is_prefilling_np.tolist() == [False, True]
    hidden = [item.input_ids[:, None].float() for item in local]
    for group in groups:
        group.gathered = hidden
    for manager, item in zip(managers, hidden):
        torch.testing.assert_close(manager.restore_hidden_states(item),
                                   batch.input_ids[:, None].float())


def test_grammar_wait_never_inserts_empty_decode_steps():
    decode = _request("decode", total=80, computed=79)
    blocked = _request("grammar", prompt=4096)
    blocked.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    blocked.structured_output_request = SimpleNamespace(grammar=None)
    scheduler = _scheduler([decode], [blocked])
    for _ in range(4):
        decode.num_computed_tokens = 79
        assert scheduler.schedule().num_scheduled_tokens == {"decode": 1}
    blocked.structured_output_request.grammar = object()
    decode.num_computed_tokens = 79
    assert "grammar" in scheduler.schedule().num_scheduled_tokens


def test_recomputing_generated_tokens_cannot_take_reserved_decode_step():
    recompute = _request("recompute", total=6160, computed=2048)
    decode = _request("decode", total=80, computed=79)
    scheduler = _scheduler([recompute, decode], [_request("new", prompt=4096)])
    setattr(scheduler, patch._ISOLATE_STREAK, 4)
    assert scheduler.schedule().num_scheduled_tokens == {"decode": 1}


def test_priority_policy_keeps_high_priority_decode_before_prefill():
    def run(isolate):
        decode = _request("decode", total=80, computed=79, priority=-10)
        prefill = _request("new", prompt=4096, priority=10)
        return _scheduler([decode], [prefill], policy=SchedulingPolicy.PRIORITY,
                          isolate=isolate).schedule().num_scheduled_tokens
    assert run(True) == run(False) == {"decode": 1, "new": 2047}


@pytest.mark.parametrize("status,grammar", [
    (RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, RuntimeError("compile failed")),
    (RequestStatus.WAITING_FOR_REMOTE_KVS, None),
    (RequestStatus.WAITING_FOR_STREAMING_REQ, None),
])
def test_blocked_waiting_prefills_do_not_defer_decodes(status, grammar):
    decode = _request("decode", total=80, computed=79)
    blocked = _request("blocked", prompt=4096)
    blocked.status = status
    blocked.structured_output_request = SimpleNamespace(grammar=grammar)
    blocked.streaming_queue = None
    scheduler = _scheduler([decode], [blocked])
    for _ in range(4):
        decode.num_computed_tokens = 79
        assert scheduler.schedule().num_scheduled_tokens == {"decode": 1}


def test_recompute_isolation_counts_generated_history_before_decode():
    recompute = _request("recompute", total=6160, computed=2048)
    decode = _request("decode", total=80, computed=79)
    scheduler = _scheduler([recompute, decode])
    assert scheduler.schedule().num_scheduled_tokens == {"recompute": 2048}
    assert decode.num_computed_tokens == 79


@pytest.mark.parametrize("short_in_skipped", [False, True])
@pytest.mark.parametrize("free_slots", [1, 2])
def test_waiting_prefill_isolation_respects_admission_slots(
    monkeypatch, short_in_skipped, free_slots,
):
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MIN_TOKENS", "128")
    decode = _request("decode", total=80, computed=79)
    short = _request("short", prompt=16)
    long = _request("long", prompt=4096)
    scheduler = _scheduler([decode], [short, long])
    scheduler.max_num_running_reqs = 1 + free_slots
    if short_in_skipped:
        scheduler.waiting.remove_request(short)
        scheduler.skipped_waiting.add_request(short)
    assert patch._pending_prefill_tokens(scheduler, 128) == (
        16 if free_slots == 1 else 4112
    )
    output = scheduler.schedule()
    assert ("decode" in output.num_scheduled_tokens) == (free_slots == 1)
    assert output.num_scheduled_tokens["short"] == 16
    assert ("long" in output.num_scheduled_tokens) == (free_slots == 2)


def test_blocked_waiter_does_not_consume_admission_slot():
    decode = _request("decode", total=80, computed=79)
    blocked = _request("blocked", prompt=4096)
    blocked.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    blocked.structured_output_request = SimpleNamespace(grammar=None)
    scheduler = _scheduler([decode], [blocked, _request("short", prompt=16),
                                       _request("long", prompt=4096)])
    scheduler.max_num_running_reqs = 2
    assert patch._pending_prefill_tokens(scheduler, 128) == 16


@pytest.mark.parametrize("queue_name", ["waiting", "skipped_waiting"])
@pytest.mark.parametrize("running_decode", [False, True])
@pytest.mark.parametrize("remote_kv", [False, True])
def test_prefill_only_step_holds_decode_ready_waiters(
    monkeypatch, queue_name, running_decode, remote_kv,
):
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MIN_TOKENS", "128")
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MAX_STEPS", "1")
    running = [_request("decode", total=80, computed=79)] if running_decode else []
    resumed = _request("resumed", total=80, computed=79)
    long = _request("long", prompt=4096)
    scheduler = _scheduler(running, [resumed, long])
    if queue_name == "skipped_waiting":
        scheduler.waiting.remove_request(resumed)
        scheduler.skipped_waiting.add_request(resumed)
    if remote_kv:
        resumed.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.finished_recving_kv_req_ids.add(resumed.request_id)

        def promote(request):
            assert request.request_id in scheduler.finished_recving_kv_req_ids
            request.status = RequestStatus.WAITING
            return True

        scheduler._try_promote_blocked_waiting_request = promote
    original_status = resumed.status
    original_queues = scheduler.waiting, scheduler.skipped_waiting
    scheduler.max_num_running_reqs = len(running) + 2
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"long": 2048}
    assert getattr(scheduler, patch._ISOLATE_STREAK) == 1
    assert (scheduler.waiting, scheduler.skipped_waiting) == original_queues
    assert list(getattr(scheduler, queue_name)) == [resumed]
    assert resumed.status == original_status
    # The reserved decode-only step admits the held waiter next time.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["resumed"] == 1
    assert "long" not in output.num_scheduled_tokens


@pytest.mark.parametrize("fail", [False, True])
def test_filtered_waiting_queues_restore_order_and_upstream_changes(monkeypatch, fail):
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MIN_TOKENS", "128")
    first = _request("first", total=80, computed=79)
    last = _request("last", total=80, computed=79)
    prefill = _request("prefill", prompt=4096)
    preempted = _request("preempted", prompt=64)
    scheduler = _scheduler([], [first, prefill, last])
    queues = scheduler.waiting, scheduler.skipped_waiting

    class ControlledScheduler:
        def schedule(self):
            assert list(self.waiting) == [prefill]
            self.waiting.pop_request()
            self.waiting.prepend_request(preempted)
            if fail:
                raise RuntimeError("schedule failed")
            return SimpleNamespace(num_scheduled_tokens={"prefill": 1})

    patch._install_pcp_prefill_isolation(ControlledScheduler)
    schedule = ControlledScheduler.schedule.__get__(scheduler)
    if fail:
        with pytest.raises(RuntimeError, match="schedule failed"):
            schedule()
    else:
        schedule()
    assert scheduler.waiting is queues[0]
    assert scheduler.skipped_waiting is queues[1]
    assert list(scheduler.waiting) == [preempted, first, last]


@pytest.mark.parametrize("queue_name", ["waiting", "skipped_waiting"])
@pytest.mark.parametrize("streaming_slots", [0, 1])
@pytest.mark.parametrize("remote_kv", [False, True])
def test_full_request_slots_do_not_reserve_waiting_decode_steps(
    monkeypatch, queue_name, streaming_slots, remote_kv,
):
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MIN_TOKENS", "128")
    monkeypatch.setenv("VLLM_HCU_PCP_ISOLATE_PREFILL_MAX_STEPS", "1")
    chunk = _request("chunk", prompt=8192, computed=2048)
    resumed = _request("resumed", total=80, computed=79)
    scheduler = _scheduler([chunk], [resumed])
    scheduler.num_waiting_for_streaming_input = streaming_slots
    scheduler.max_num_running_reqs = 1 + streaming_slots
    if queue_name == "skipped_waiting":
        scheduler.waiting.remove_request(resumed)
        scheduler.skipped_waiting.add_request(resumed)
    if remote_kv:
        resumed.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.finished_recving_kv_req_ids.add(resumed.request_id)

        def promote(request):
            assert request.request_id in scheduler.finished_recving_kv_req_ids
            request.status = RequestStatus.WAITING
            return True

        scheduler._try_promote_blocked_waiting_request = promote

    # Even a streak inherited from a previous mixed phase must not reserve an
    # empty decode step while the waiting request cannot be admitted.
    setattr(scheduler, patch._ISOLATE_STREAK, 1)
    for _ in range(4):
        # Keep the fixture in the chunked-prefill phase across repeated calls.
        chunk.num_computed_tokens = 2048
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"chunk": 2048}
        assert getattr(scheduler, patch._ISOLATE_STREAK) == 0
        assert list(getattr(scheduler, queue_name)) == [resumed]

    # Once a slot opens, isolation and its reserved decode step work again.
    scheduler.max_num_running_reqs += 1
    chunk.num_computed_tokens = 2048
    assert scheduler.schedule().num_scheduled_tokens == {"chunk": 2048}
    assert scheduler.schedule().num_scheduled_tokens == {"resumed": 1}
