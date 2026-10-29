# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


def fingerprint(key, config, tp_size, tp_rank):
    from vllm_hcu.runtime_compat.weight_cache import _weight_cache_config_scope

    vllm_config = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="auto"))
    with _weight_cache_config_scope(vllm_config):
        return key.from_model_config(config, tp_size, tp_rank)


@pytest.mark.parametrize("mode", ["short", "underscore", "yaml", "tp_only"])
def test_launcher_selects_pipeline_using_engine_args(mode, tmp_path, monkeypatch):
    import sys
    from vllm.engine.arg_utils import EngineArgs
    from vllm_hcu.preload.pp_daemon import main

    if mode == "yaml":
        path = tmp_path / "engine.yaml"
        path.write_text("pipeline-parallel-size: 2\n")
        options = ["--config", str(path)]
    elif mode == "short":
        options = ["-pp=2"]
    elif mode == "underscore":
        options = ["--pipeline_parallel_size", "2"]
    else:
        options = ["--pipeline-parallel-size", "1"]
    monkeypatch.setattr(sys, "argv", ["preload", *options])

    def fake_create(self):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=self.pipeline_parallel_size,
            ),
            load_config=SimpleNamespace(load_format="ipc_cache"),
        )

    monkeypatch.setattr(EngineArgs, "create_engine_config", fake_create)
    calls = []
    upstream = SimpleNamespace(main=lambda: calls.append("tp"))
    if mode == "tp_only":
        main(upstream)
        assert calls == ["tp"]
    else:
        with pytest.raises(ValueError, match="must load from disk"):
            main(upstream)
        assert not calls


def test_startup_signal_during_queue_wait_is_not_a_failure():
    import queue
    from vllm_hcu.preload.pp_daemon import _wait_until_ready

    state = {"stopping": False, "joined": False}

    class ReadyQueue:
        def get(self, timeout):
            state["stopping"] = True
            raise queue.Empty

    class Process:
        exitcode = -15

        def join(self):
            state["joined"] = True

    assert not _wait_until_ready([Process()], ReadyQueue(), 1, lambda: state["stopping"])
    assert state["joined"]


def test_daemon_monitor_stops_survivors_when_child_exits():
    import multiprocessing
    import time

    from vllm_hcu.preload.pp_daemon import _monitor_processes

    ctx = multiprocessing.get_context("spawn")
    survivor = ctx.Process(target=time.sleep, args=(30,), name="survivor")
    failed = ctx.Process(target=time.sleep, args=(0,), name="exited")
    survivor.start()
    failed.start()
    try:
        with pytest.raises(SystemExit, match="exited unexpectedly: exited"):
            _monitor_processes([survivor, failed], lambda: False)
        assert survivor.exitcode is not None
    finally:
        for process in (survivor, failed):
            if process.is_alive():
                process.terminate()
            process.join()


def test_pipeline_rank_plan_keeps_stage_and_tp_rank_isolated():
    from vllm_hcu.preload.pp_daemon import plan_local_pipeline_ranks

    node0 = plan_local_pipeline_ranks(8, 2, 2, 0)
    node1 = plan_local_pipeline_ranks(8, 2, 2, 1)
    assert node0[0] == (0, 0, 0, 0)
    assert node0[-1] == (7, 7, 0, 7)
    assert node1[0] == (8, 0, 1, 0)
    assert node1[-1] == (15, 7, 1, 7)


def test_pipeline_rank_plan_rejects_unbalanced_nodes():
    from vllm_hcu.preload.pp_daemon import plan_local_pipeline_ranks

    with pytest.raises(ValueError, match="divisible"):
        plan_local_pipeline_ranks(3, 2, 4, 0)


def test_pipeline_socket_scope_uses_stage(monkeypatch):
    from vllm_hcu.runtime_compat.weight_cache import (
        _pipeline_context,
        _pp_socket_identifier,
    )

    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_SIZE", "2")
    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_RANK", "1")
    assert _pipeline_context() == (2, 1)
    assert _pp_socket_identifier("uuid-GPU-1") == "uuid-GPU-1-pp1"

    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_SIZE", "1")
    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_RANK", "0")
    assert _pp_socket_identifier("uuid-GPU-1") == "uuid-GPU-1"


def test_pipeline_cache_key_includes_stage(monkeypatch, tmp_path):
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("base", "vllm")

    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config = SimpleNamespace(
        model=str(tmp_path), hf_config=SimpleNamespace(to_dict=lambda: {}),
        model_class_overrides={},
    )
    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_SIZE", "2")
    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_RANK", "0")
    first = fingerprint(Key, config, 8, 0)
    monkeypatch.setenv("VLLM_HCU_PRELOAD_PP_RANK", "1")
    second = fingerprint(Key, config, 8, 0)
    assert first.checkpoint != second.checkpoint
    assert "hcu-pp:2:0" in first.checkpoint
    assert "hcu-pp:2:1" in second.checkpoint


def test_slimquant_ipc_is_allowed_without_extra_configuration():
    from vllm_hcu.runtime_compat.weight_cache import install_ipc_quant_support

    module = SimpleNamespace(IPC_QUANT_ALLOWLIST={None: lambda _: True})
    install_ipc_quant_support(module)
    assert module.IPC_QUANT_ALLOWLIST["slimquant_w4a8"](None)


def test_slimquant_ipc_gate_is_idempotent():
    from vllm_hcu.runtime_compat.weight_cache import install_ipc_quant_support

    module = SimpleNamespace(IPC_QUANT_ALLOWLIST={None: lambda _: True})
    install_ipc_quant_support(module)
    predicate = module.IPC_QUANT_ALLOWLIST["slimquant_w4a8"]
    install_ipc_quant_support(module)
    assert module.IPC_QUANT_ALLOWLIST["slimquant_w4a8"] is predicate


def test_slimquant_ipc_overrides_preexisting_entry():
    from vllm_hcu.runtime_compat.weight_cache import install_ipc_quant_support

    module = SimpleNamespace(IPC_QUANT_ALLOWLIST={
        None: lambda _: True,
        "slimquant_w4a8": lambda _: False,
    })
    install_ipc_quant_support(module)
    assert module.IPC_QUANT_ALLOWLIST["slimquant_w4a8"](None)
