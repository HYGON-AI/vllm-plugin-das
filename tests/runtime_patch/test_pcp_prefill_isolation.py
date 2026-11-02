# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek V4 PCP: long prefills are scheduled without concurrent decodes."""

from types import SimpleNamespace

from vllm.v1.core.sched.request_queue import FCFSRequestQueue

from vllm_hcu.patch.platform.framework_opt import patch_scheduler


def _request(name, prompt, computed, eligible=0):
    return SimpleNamespace(
        request_id=name,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        next_decode_eligible_step=eligible,
    )


class _FakeScheduler:
    """Mimics the upstream eligibility gate and a KV-limited waiting queue."""

    def __init__(self, running, waiting, kv_room=True, max_running=64):
        self.current_step = 0
        self.max_num_running_reqs = max_running
        self.running = running
        self.waiting = FCFSRequestQueue(waiting)
        self.skipped_waiting = FCFSRequestQueue()
        self.kv_room = kv_room
        self.preempt_on_next_step = None
        setattr(self, patch_scheduler._ISOLATE_PREFILL, True)

    def schedule(self):
        self.current_step += 1
        scheduled = {
            r.request_id: 1
            for r in self.running
            if self.current_step >= r.next_decode_eligible_step
        }
        if self.kv_room:
            for r in self.waiting:
                scheduled[r.request_id] = r.num_prompt_tokens
        if self.preempt_on_next_step is not None:
            self.waiting.prepend_request(self.preempt_on_next_step)
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


def test_decode_only_step_keeps_requests_preempted_meanwhile(monkeypatch):
    sched, _ = _prefill_burst(monkeypatch, 1)
    setattr(sched, patch_scheduler._ISOLATE_STREAK, 1)
    victim = _request("victim", 500, 0)
    sched.preempt_on_next_step = victim
    out = sched.schedule()
    assert set(out.num_scheduled_tokens) == {"decode"}
    queued = [r.request_id for r in sched.waiting]
    assert queued[0] == "victim" and len(queued) == 9


def test_unlimited_streak_never_forces_decode_step(monkeypatch):
    sched, _ = _prefill_burst(monkeypatch, 0)
    setattr(sched, patch_scheduler._ISOLATE_STREAK, 100)
    assert "decode" not in sched.schedule().num_scheduled_tokens
