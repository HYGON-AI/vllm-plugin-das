# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek V4 PCP: long prefills are scheduled without concurrent decodes."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import RequestStatus

from vllm_hcu.patch.platform.framework_opt import patch_scheduler


def _request(name, prompt, computed, eligible=0):
    return SimpleNamespace(
        request_id=name,
        num_prompt_tokens=prompt,
        num_tokens=max(prompt, computed + 1),
        status=RequestStatus.WAITING,
        num_stale_output_tokens=0,
        drop_stale_output=False,
        num_computed_tokens=computed,
        next_decode_eligible_step=eligible,
    )


class _FakeScheduler:
    """Mimics the upstream eligibility gate and a KV-limited waiting queue."""

    def __init__(self, running, waiting, kv_room=True, max_running=64):
        self.policy = SchedulingPolicy.FCFS
        self.current_step = 0
        self._pause_state = PauseState.UNPAUSED
        self.num_waiting_for_streaming_input = 0
        self.max_num_running_reqs = max_running
        self.running = running
        self.waiting = FCFSRequestQueue(waiting)
        self.skipped_waiting = FCFSRequestQueue()
        self.kv_room = kv_room
        self.preempt_on_next_step = None
        setattr(self, patch_scheduler._ISOLATE_PREFILL, True)

    def schedule(self):
        self.current_step += 1
        if self._pause_state == PauseState.PAUSED_ALL:
            return SimpleNamespace(num_scheduled_tokens={})
        scheduled = {
            r.request_id: 1
            for r in self.running
            if self.current_step >= r.next_decode_eligible_step
        }
        if (
            self.kv_room
            and self._pause_state == PauseState.UNPAUSED
            and len(self.running) + self.num_waiting_for_streaming_input
            < self.max_num_running_reqs
        ):
            for r in self.waiting:
                scheduled[r.request_id] = r.num_prompt_tokens
        if self.preempt_on_next_step is not None:
            for request in self.preempt_on_next_step:
                self.waiting.prepend_request(request)
            self.preempt_on_next_step = None
        return SimpleNamespace(num_scheduled_tokens=scheduled)


patch_scheduler._install_pcp_prefill_isolation(_FakeScheduler)


def test_long_prefill_step_defers_decodes_once():
    decode = _request("decode", 100, 120)
    sched = _FakeScheduler([decode], [_request("new", 12288, 0)])
    first = sched.schedule()
    assert set(first.num_scheduled_tokens) == {"new"}
    assert decode.next_decode_eligible_step == 0
    sched.waiting = []
    assert set(sched.schedule().num_scheduled_tokens) == {"decode"}


def test_short_prefill_still_mixes_with_decode():
    sched = _FakeScheduler([_request("decode", 100, 120)], [_request("new", 64, 0)])
    assert set(sched.schedule().num_scheduled_tokens) == {"decode", "new"}


def test_unschedulable_prefill_backs_off_so_decode_progresses():
    sched = _FakeScheduler(
        [_request("decode", 100, 120)], [_request("new", 12288, 0)], kv_room=False
    )
    assert sched.schedule().num_scheduled_tokens == {}
    assert set(sched.schedule().num_scheduled_tokens) == {"decode"}
    # Back-off lasts one step only.
    assert sched.schedule().num_scheduled_tokens == {}


def test_running_chunked_prefill_is_kept_and_decode_deferred():
    chunk = _request("chunk", 40960, 32768)
    decode = _request("decode", 100, 120)
    sched = _FakeScheduler([chunk, decode], [])
    assert set(sched.schedule().num_scheduled_tokens) == {"chunk"}


def test_existing_eligibility_gate_is_preserved():
    decode = _request("decode", 100, 120, eligible=5)
    sched = _FakeScheduler([decode], [_request("new", 12288, 0)])
    sched.schedule()
    assert decode.next_decode_eligible_step == 5


def test_disabled_isolation_keeps_upstream_schedule():
    sched = _FakeScheduler([_request("decode", 100, 120)], [_request("new", 12288, 0)])
    setattr(sched, patch_scheduler._ISOLATE_PREFILL, False)
    assert set(sched.schedule().num_scheduled_tokens) == {"decode", "new"}


def test_full_running_set_does_not_defer_decodes():
    sched = _FakeScheduler(
        [_request("decode", 100, 120)], [_request("new", 12288, 0)], max_running=1
    )
    assert "decode" in sched.schedule().num_scheduled_tokens


def _prefill_burst(monkeypatch, max_steps):
    import vllm_hcu.platforms.envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_PCP_ISOLATE_PREFILL_MAX_STEPS", max_steps,
                        raising=False)
    waiting = [_request(f"new{i}", 12288, 0) for i in range(8)]
    sched = _FakeScheduler([_request("decode", 100, 120)], waiting)

    def step():
        out = sched.schedule()
        # Scheduled prefills leave the queue as the real scheduler would.
        for name in out.num_scheduled_tokens:
            for r in list(sched.waiting):
                if r.request_id == name:
                    sched.waiting.remove_request(r)
                    break
        return set(out.num_scheduled_tokens)

    return sched, step


def test_prefill_burst_inserts_decode_only_step(monkeypatch):
    sched, step = _prefill_burst(monkeypatch, 2)
    seen = []
    # Admit one prefill per step, like a long prompt filling the token budget.
    for _ in range(6):
        head = list(sched.waiting)[:1]
        rest = list(sched.waiting)[1:]
        sched.waiting = FCFSRequestQueue(head)
        names = step()
        for r in rest:
            sched.waiting.add_request(r)
        seen.append(names)
    assert seen[:3] == [{"new0"}, {"new1"}, {"decode"}]
    assert seen[3:] == [{"new2"}, {"new3"}, {"decode"}]


@pytest.mark.parametrize("num_preemptions", [1, 2])
def test_decode_only_step_keeps_requests_preempted_meanwhile(monkeypatch, num_preemptions):
    sched, _ = _prefill_burst(monkeypatch, 1)
    setattr(sched, patch_scheduler._ISOLATE_STREAK, 1)
    victims = [_request(f"victim{i}", 500, 0) for i in range(num_preemptions)]
    expected = FCFSRequestQueue(sched.waiting)
    for victim in victims:
        expected.prepend_request(victim)
    original_waiting = sched.waiting
    sched.preempt_on_next_step = victims
    out = sched.schedule()
    assert set(out.num_scheduled_tokens) == {"decode"}
    queued = [r.request_id for r in sched.waiting]
    assert sched.waiting is original_waiting
    assert queued == [r.request_id for r in expected]


def test_unlimited_streak_never_forces_decode_step(monkeypatch):
    sched, _ = _prefill_burst(monkeypatch, 0)
    setattr(sched, patch_scheduler._ISOLATE_STREAK, 100)
    assert "decode" not in sched.schedule().num_scheduled_tokens

@pytest.mark.parametrize("enabled", [False, True])
def test_custom_ops_master_controls_isolation(monkeypatch, enabled):
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", str(int(enabled)))
    monkeypatch.setattr(henvs, "VLLM_HCU_PCP_ISOLATE_PREFILL", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES", True)
    sched = _FakeScheduler(
        [_request("decode", 100, 120)], [_request("new", 12288, 0)]
    )
    delattr(sched, patch_scheduler._ISOLATE_PREFILL)
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=8),
    )
    expected = {"new"} if enabled else {"decode", "new"}
    assert set(sched.schedule().num_scheduled_tokens) == expected
    assert getattr(sched, patch_scheduler._ISOLATE_PREFILL) is enabled


@pytest.mark.parametrize("queue_name", ["waiting", "skipped_waiting"])
def test_paused_new_drains_decodes_without_empty_steps(queue_name):
    decode = _request("decode", 100, 120)
    waiting = _request("new", 4096, 0)
    sched = _FakeScheduler([decode], [])
    getattr(sched, queue_name).add_request(waiting)
    sched._pause_state = PauseState.PAUSED_NEW

    for _ in range(4):
        assert set(sched.schedule().num_scheduled_tokens) == {"decode"}
        assert decode.next_decode_eligible_step == 0
    assert list(getattr(sched, queue_name)) == [waiting]
    # Resuming admission must immediately allow prefill isolation again.
    sched._pause_state = PauseState.UNPAUSED
    # The fake scheduler only implements admission from the waiting queue.
    if queue_name == "skipped_waiting":
        sched.waiting.add_request(sched.skipped_waiting.pop_request())
    assert set(sched.schedule().num_scheduled_tokens) == {"new"}


def test_paused_new_still_isolates_running_chunked_prefill():
    sched = _FakeScheduler(
        [_request("chunk", 8192, 0), _request("decode", 100, 120)],
        [_request("new", 4096, 0)],
    )
    sched._pause_state = PauseState.PAUSED_NEW
    assert set(sched.schedule().num_scheduled_tokens) == {"chunk"}


def test_paused_all_does_not_defer_or_count_prefills():
    decode = _request("decode", 100, 120)
    sched = _FakeScheduler([decode], [_request("new", 4096, 0)])
    sched._pause_state = PauseState.PAUSED_ALL
    assert patch_scheduler._pending_prefill_tokens(sched, 1024) == 0
    assert sched.schedule().num_scheduled_tokens == {}
    assert decode.next_decode_eligible_step == 0


def test_streaming_sessions_filling_running_slots_do_not_defer_decodes():
    sched = _FakeScheduler(
        [_request("decode", 100, 120)], [_request("new", 4096, 0)], max_running=2
    )
    sched.num_waiting_for_streaming_input = 1
    assert "decode" in sched.schedule().num_scheduled_tokens
