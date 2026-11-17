"""Real child-process tests for post-READY failure and bounded shutdown."""
import multiprocessing
import os
import time
import pytest
from hcu_vllm_das.parallel import minimax31_preload_cli as cli


def test_later_rank_death_is_not_hidden_by_first_live_daemon():
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=time.sleep, args=(60,), name="live-first"),
             ctx.Process(target=os._exit, args=(7,), name="failed-later")]
    try:
        for proc in procs:
            proc.start()
        with pytest.raises(RuntimeError, match="after readiness.*failed-later"):
            cli._wait_for_workers(procs, lambda: False)
    finally:
        cli._stop_workers(procs, grace_s=1)
    assert all(not proc.is_alive() for proc in procs)


def test_shutdown_kills_unresponsive_worker_and_ignores_unstarted_process():
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=exec, args=(
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); ready.set(); time.sleep(60)",
        {"ready": ready},))
    procs = [proc, ctx.Process(target=time.sleep, args=(60,))]
    proc.start()
    try:
        assert ready.wait(10)
        cli._stop_workers(procs, grace_s=.1)
        assert not proc.is_alive()
        assert proc.exitcode is not None
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)


def test_requested_shutdown_returns_without_waiting_on_sentinels():
    assert cli._wait_for_workers([], lambda: True) == 0
