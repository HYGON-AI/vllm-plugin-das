# SPDX-License-Identifier: Apache-2.0
"""VLLM_HCU_PREFILL_BATCH_*: hold new prefills while decodes run."""

import time
import types

import pytest

from vllm_hcu.patch.platform.framework_opt import patch_scheduler


class _Queue(list):
    def peek_request(self):
        return self[0]


def _make(monkeypatch, wait_ms, min_batch, min_running=0):
    import vllm_hcu.platforms.envs as henvs

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")

    monkeypatch.setattr(henvs, "VLLM_HCU_PREFILL_BATCH_WAIT_MS", wait_ms, raising=False)
    monkeypatch.setattr(henvs, "VLLM_HCU_PREFILL_BATCH_MIN", min_batch, raising=False)
    monkeypatch.setattr(henvs, "VLLM_HCU_PREFILL_BATCH_MIN_RUNNING", min_running, raising=False)

    class Scheduler:
        def schedule(self, throttle_prefills: bool = False):
            return throttle_prefills

    patch_scheduler._install_prefill_batching(Scheduler)
    return Scheduler()


def _req(age_ms):
    return types.SimpleNamespace(arrival_time=time.time() - age_ms / 1000.0)


@pytest.mark.parametrize(
    "wait_ms,min_batch,running,waiting_ages_ms,expected",
    [
        (0, 0, [1], [0], False),          # disabled
        (50, 0, [1], [0], True),          # fresh arrival is held
        (50, 0, [1], [100], False),       # oldest waited long enough
        (50, 2, [1], [0, 0], False),      # MIN reached
        (50, 3, [1], [0, 0], True),       # MIN not reached
        (50, 0, [], [0], False),          # nothing decoding: never hold
        (50, 0, [1], [], False),          # nothing waiting
    ],
)
def test_prefill_batching(monkeypatch, wait_ms, min_batch, running,
                          waiting_ages_ms, expected):
    s = _make(monkeypatch, wait_ms, min_batch)
    s.running = running
    # Build requests here, not at collection time, so their age is exact.
    s.waiting = _Queue([_req(age) for age in waiting_ages_ms])
    s.skipped_waiting = _Queue()
    assert s.schedule() is expected


def test_upstream_throttle_is_kept(monkeypatch):
    s = _make(monkeypatch, 50, 0)
    s.running, s.waiting, s.skipped_waiting = [1], _Queue([_req(100)]), _Queue()
    assert s.schedule(True) is True


def test_skips_schedule_without_throttle_param():
    class Scheduler:
        def schedule(self):
            return "orig"

    orig = Scheduler.schedule
    patch_scheduler._install_prefill_batching(Scheduler)
    assert Scheduler.schedule is orig


@pytest.mark.parametrize("num_running,expected", [(3, False), (8, True)])
def test_min_running(monkeypatch, num_running, expected):
    s = _make(monkeypatch, 50, 0, min_running=8)
    s.running = list(range(num_running))
    s.waiting, s.skipped_waiting = _Queue([_req(0)]), _Queue()
    assert s.schedule() is expected


def test_master_switch_off_disables(monkeypatch):
    s = _make(monkeypatch, 50, 0)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    s.running, s.waiting, s.skipped_waiting = [1], _Queue([_req(0)]), _Queue()
    assert s.schedule() is False


@pytest.mark.parametrize("master,expected_none", [("0", True), ("1", False)])
def test_cache_dequant_follows_master_switch(monkeypatch, master, expected_none):
    import torch

    import vllm_hcu.platforms.envs as henvs
    from vllm_hcu.model_executor.layers.quantization import w8a16_gemm

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setattr(henvs, "VLLM_HCU_W8A16_CACHE_DEQUANT", True, raising=False)
    sentinel = torch.empty(1)
    monkeypatch.setattr(w8a16_gemm, "_w8a16_dequant", lambda *a: sentinel)
    monkeypatch.setattr(w8a16_gemm, "_DEQUANT_CACHE", {})
    w = torch.zeros(16, 128, dtype=torch.int8)
    out = w8a16_gemm._cached_dequant(w, None, torch.float16, 0)
    assert (out is None) is expected_none


def test_defaults(monkeypatch):
    """int8 lm_head and prefill batching are opt-in (they change numerics /
    trade TTFT for throughput on every model); the other Qwen3.5
    optimizations are on."""
    import vllm_hcu.platforms.envs as henvs

    names = (
        "VLLM_HCU_INT8_LM_HEAD",
        "VLLM_HCU_USE_FUSED_QWEN35_MOE_BLOCK",
        "VLLM_HCU_FUSED_QWEN35_MOE_NORM",
        "VLLM_HCU_W8A16_CACHE_DEQUANT",
        "VLLM_HCU_PREFILL_BATCH_WAIT_MS",
        "VLLM_HCU_PREFILL_BATCH_MIN",
        "VLLM_HCU_PREFILL_BATCH_MIN_RUNNING",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    defaults = {n: henvs.hcu_vllm_environment_variables[n]() for n in names}
    assert defaults == {
        "VLLM_HCU_INT8_LM_HEAD": False,
        "VLLM_HCU_USE_FUSED_QWEN35_MOE_BLOCK": True,
        "VLLM_HCU_FUSED_QWEN35_MOE_NORM": True,
        "VLLM_HCU_W8A16_CACHE_DEQUANT": True,
        "VLLM_HCU_PREFILL_BATCH_WAIT_MS": 0,
        "VLLM_HCU_PREFILL_BATCH_MIN": 0,
        "VLLM_HCU_PREFILL_BATCH_MIN_RUNNING": 12,
    }
