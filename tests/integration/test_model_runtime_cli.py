# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-safe CLI coverage for the model runtime subprocess harness."""

from pathlib import Path
import copy
import json
import signal
import sys
from types import SimpleNamespace

import pytest

from tests.integration import model_runtime


DEEPSEEK_V4_MODEL = Path("/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8")
MTP3_SELECTOR_NAMES = (
    "VLLM_HCU_USE_CUSTOM_OPS",
    "VLLM_HCU_USE_CUSTOM_AITER_FLA",
    "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D",
    "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
    "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP",
    "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O",
)


@pytest.mark.parametrize("resolved_mode", ["FULL", "FULL_DECODE_ONLY"])
def test_mtp3_graph_cli_uses_worker_config_and_repeats_sequential_engines(
    monkeypatch, capsys, resolved_mode,
):
    events = []
    engines = []
    counter = SimpleNamespace(num_cudagraph_captured=0)
    for name in MTP3_SELECTOR_NAMES:
        monkeypatch.delenv(name, raising=False)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.label = "eager" if kwargs["enforce_eager"] else "graph"
            self.round = 0
            engines.append(self)
            events.append((self.label, "start"))
            # Deliberately differs from the resolved worker config.
            self.llm_engine = SimpleNamespace(
                vllm_config=None,
                do_log_stats=lambda: events.append((self.label, "flush")),
            )

        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert use_tqdm is False
            self.round += 1
            events.append((self.label, "generate"))
            self.prompts = list(prompts)
            self.sampling = sampling_params
            wants_logprobs = getattr(sampling_params, "logprobs", None) == 1
            return [SimpleNamespace(
                prompt_token_ids=[5, 6],
                outputs=[SimpleNamespace(
                    token_ids=list(range(16)), text="sixteen tokens",
                    finish_reason="length",
                    cumulative_logprob=-0.5 if wants_logprobs else None,
                    logprobs=[{i: -0.1} for i in range(16)] if wants_logprobs else None,
                )],
            ) for _ in prompts]

        def collective_rpc(self, method, *, timeout):
            assert timeout == 30
            mode = "NONE" if self.label == "eager" else resolved_mode
            counter.num_cudagraph_captured = 0 if self.label == "eager" else 2
            worker = SimpleNamespace(vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=self.label == "eager"),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=2, data_parallel_size=1,
                    enable_expert_parallel=True,
                ),
                speculative_config=SimpleNamespace(
                    method="mtp", num_speculative_tokens=3,
                ),
                compilation_config=SimpleNamespace(
                    mode=SimpleNamespace(
                        name="NONE" if self.label == "eager" else "VLLM_COMPILE"
                    ),
                    cudagraph_mode=SimpleNamespace(
                        name=mode,
                        decode_mode=lambda: SimpleNamespace(
                            name="NONE" if self.label == "eager" else "FULL"
                        ),
                    ),
                    cudagraph_capture_sizes=[] if self.label == "eager" else [4, 8],
                ),
            ))
            return [method(worker), method(worker)]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setitem(
        sys.modules, "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.compilation.counter",
        SimpleNamespace(compilation_counter=counter),
    )
    monkeypatch.setattr(
        model_runtime, "_shutdown_llm",
        lambda llm: events.append((llm.label, "stop")),
    )
    assert model_runtime._main([
        "qwen35-mtp3-graph-parity", "--model", "/models/fake",
        "--gpu-memory-utilization", "0.4",
    ]) == 0
    log = capsys.readouterr().out
    result = json.loads(next(
        line.removeprefix(model_runtime.RESULT_PREFIX)
        for line in log.splitlines() if line.startswith(model_runtime.RESULT_PREFIX)
    ))
    assert events == [
        (label, event) for label in ("eager", "graph")
        for event in ("start", "generate", "flush", "generate", "flush", "stop")
    ]
    for engine in engines:
        assert engine.kwargs["speculative_config"] == {
            "method": "mtp", "num_speculative_tokens": 3,
        }
        assert engine.kwargs["tensor_parallel_size"] == 2
        assert engine.kwargs["enable_expert_parallel"] is True
        assert engine.kwargs["moe_backend"] == "aiter"
        assert engine.kwargs["gpu_memory_utilization"] == 0.4
        assert engine.kwargs["cudagraph_metrics"] is True
        assert engine.kwargs["disable_log_stats"] is False
        assert engine.sampling.temperature == 0.0
        assert engine.sampling.seed == 0
        assert engine.sampling.max_tokens == 16
        assert engine.sampling.min_tokens == 16
        assert engine.sampling.ignore_eos is True
        assert getattr(engine.sampling, "logprobs", None) == 1
        workers = result[engine.label]["workers"]
        assert len(workers) == 2
        assert all(
            worker.get("hcu_selectors") == {
                name: True for name in MTP3_SELECTOR_NAMES
            }
            for worker in workers
        )
        assert all(
            w["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3}
            for w in workers
        )
        assert len(result[engine.label]["rounds"]) == 2
        for output in result[engine.label]["rounds"]:
            assert all(record["cumulative_logprob"] == -0.5 for record in output)
            assert all(record["sample_logprob_count"] == 16 for record in output)
    assert engines[0].prompts == engines[1].prompts
    assert engines[0].kwargs.get("compilation_config") is None
    assert engines[1].kwargs["compilation_config"] == {
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8],
    }
    assert result["graph"]["workers"][0]["compilation_config"] == {
        "mode": "VLLM_COMPILE", "cudagraph_mode": resolved_mode,
        "decode_mode": "FULL", "cudagraph_capture_sizes": [4, 8],
        "num_cudagraph_captured": 2,
    }
    for label in ("eager", "graph"):
        for round_index in range(2):
            assert f"VLLM_HCU_GENERATE_BEGIN={label}:{round_index}" in log
            assert f"VLLM_HCU_GENERATE_END={label}:{round_index}" in log


def _mtp3_parity_payload():
    worker = {
        "enforce_eager": False,
        "hcu_selectors": {name: True for name in MTP3_SELECTOR_NAMES},
        "parallel_config": {
            "tensor_parallel_size": 2, "data_parallel_size": 1,
            "enable_expert_parallel": True,
        },
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        "compilation_config": {
            "mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_mode": "FULL", "cudagraph_capture_sizes": [4, 8],
            "num_cudagraph_captured": 2,
        },
    }
    result = {
        label: {"workers": [copy.deepcopy(worker), copy.deepcopy(worker)],
                "rounds": [[{
                    "token_ids": list(range(16)), "cumulative_logprob": -0.5,
                    "sample_logprob_count": 16,
                }] for _ in range(2)]}
        for label in ("eager", "graph")
    }
    for entry in result["eager"]["workers"]:
        entry["enforce_eager"] = True
        entry["compilation_config"].update(
            mode="NONE", cudagraph_mode="NONE", decode_mode="NONE",
            num_cudagraph_captured=0,
        )
    return result


def _mtp3_parity_log():
    return "\n".join(
        f"VLLM_HCU_GENERATE_BEGIN={label}:{i}\n"
        "SpecDecoding metrics: Accepted: 8 tokens, Drafted: 12 tokens,\n"
        f"| 8 | 8 | 0 | {'FULL' if label == 'graph' else 'NONE'} | 4 |\n"
        f"VLLM_HCU_GENERATE_END={label}:{i}"
        for label in ("eager", "graph") for i in range(2)
    )


def test_mtp3_graph_callable_rpc_opt_in_is_local_and_logged(monkeypatch, tmp_path):
    from tests.integration.graph import test_qwen35_35b_a3b_mtp3_graph_parity as case

    launched = []
    opt_in = "VLLM_ALLOW_INSECURE_SERIALIZATION"
    monkeypatch.delenv(opt_in, raising=False)
    for name in MTP3_SELECTOR_NAMES:
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("VLLM_HCU_INTEGRATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(case, "require_gfx_arch", lambda *args: None)
    monkeypatch.setattr(
        case, "require_model_runtime", lambda *args, **kwargs: Path("/models/fake")
    )

    def fake_popen(command, *, env, stdout, **kwargs):
        launched.append((dict(env), Path(stdout.name)))
        stdout.write(_mtp3_parity_log() + "\n")
        stdout.write(
            model_runtime.RESULT_PREFIX + json.dumps(_mtp3_parity_payload()) + "\n"
        )
        stdout.flush()
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr(model_runtime.subprocess, "Popen", fake_popen)
    with monkeypatch.context() as case_env:
        case.test_qwen35_35b_a3b_tp2_ep2_mtp3_full_decode_graph_aiter_auto_shuffle(
            None, case_env,
        )
    model_runtime.run_vllm_case("tp-ep-smoke", Path("/models/fake"))

    targeted_env, targeted_log = launched[0]
    smoke_env, smoke_log = launched[1]
    targeted_opt_in = targeted_env.get(opt_in)
    assert targeted_opt_in == "1"
    environment_header = next(
        line for line in targeted_log.read_text().splitlines()
        if line.startswith("environment: ")
    )
    assert f"{opt_in}=1" in environment_header
    assert opt_in not in smoke_env
    assert opt_in not in smoke_log.read_text()
    assert opt_in not in model_runtime.os.environ
    inherited_selectors = set(MTP3_SELECTOR_NAMES) & targeted_env.keys()
    assert not inherited_selectors
    assert all(smoke_env[name] == "0" for name in MTP3_SELECTOR_NAMES)
    assert all(model_runtime.os.environ[name] == "0" for name in MTP3_SELECTOR_NAMES)


@pytest.mark.parametrize("selector", MTP3_SELECTOR_NAMES)
def test_mtp3_graph_rejects_disabled_worker_selector(selector):
    from tests.integration.graph.test_qwen35_35b_a3b_mtp3_graph_parity import (
        _assert_mtp3_graph_parity,
    )

    result = _mtp3_parity_payload()
    result["graph"]["workers"][1]["hcu_selectors"][selector] = False
    with pytest.raises(AssertionError):
        _assert_mtp3_graph_parity(result, _mtp3_parity_log())


@pytest.mark.parametrize("graph_logprob", [-0.75, float("nan"), float("inf")])
def test_mtp3_graph_logprobs_require_finiteness_not_numerical_parity(graph_logprob):
    from tests.integration.graph.test_qwen35_35b_a3b_mtp3_graph_parity import (
        _assert_mtp3_graph_parity,
    )

    result = _mtp3_parity_payload()
    for output in result["graph"]["rounds"]:
        output[0]["cumulative_logprob"] = graph_logprob
    if graph_logprob == -0.75:
        # The eager cumulative value is -0.5 with the same exact token IDs.
        _assert_mtp3_graph_parity(result, _mtp3_parity_log())
    else:
        with pytest.raises(AssertionError):
            _assert_mtp3_graph_parity(result, _mtp3_parity_log())


@pytest.mark.parametrize("fault", [
    None, "downgraded", "uncaptured", "empty_capture_sizes", "wrong_mtp",
    "wrong_draft_count", "wrong_tp", "missing_worker", "empty_round",
    "short_output", "token_mismatch", "second_round_mismatch", "missing_full_metric",
    "zero_full_metric", "missing_drafts", "zero_drafts",
    "missing_logprobs", "short_logprobs",
])
def test_mtp3_graph_assertions_reject_false_parity(fault):
    from tests.integration.graph.test_qwen35_35b_a3b_mtp3_graph_parity import (
        _assert_mtp3_graph_parity,
    )
    result = _mtp3_parity_payload()
    log = _mtp3_parity_log()
    worker = result["graph"]["workers"][1]
    if fault == "downgraded":
        worker["compilation_config"]["decode_mode"] = "PIECEWISE"
    elif fault == "uncaptured":
        worker["compilation_config"]["num_cudagraph_captured"] = 0
    elif fault == "empty_capture_sizes":
        worker["compilation_config"]["cudagraph_capture_sizes"] = []
    elif fault == "wrong_mtp":
        worker["speculative_config"]["method"] = "eagle"
    elif fault == "wrong_draft_count":
        worker["speculative_config"]["num_speculative_tokens"] = 1
    elif fault == "wrong_tp":
        worker["parallel_config"]["tensor_parallel_size"] = 1
    elif fault == "missing_worker":
        result["graph"]["workers"].pop()
    elif fault == "empty_round":
        result["graph"]["rounds"][1] = []
    elif fault == "short_output":
        result["graph"]["rounds"][1][0]["token_ids"] = [1]
    elif fault == "token_mismatch":
        result["graph"]["rounds"][0][0]["token_ids"][0] = 99
    elif fault == "second_round_mismatch":
        for label in ("eager", "graph"):
            result[label]["rounds"][1][0]["token_ids"][0] = 99
    elif fault == "missing_full_metric":
        log = log.replace("| FULL |", "| NONE |")
    elif fault == "zero_full_metric":
        log = log.replace("| FULL | 4 |", "| FULL | 0 |")
    elif fault == "missing_drafts":
        log = log.replace("SpecDecoding metrics:", "unrelated:")
    elif fault == "zero_drafts":
        log = log.replace("Drafted: 12", "Drafted: 0")
    elif fault == "missing_logprobs":
        result["graph"]["rounds"][1][0]["cumulative_logprob"] = None
    elif fault == "short_logprobs":
        result["graph"]["rounds"][1][0]["sample_logprob_count"] = 15
    if fault is None:
        _assert_mtp3_graph_parity(result, log)
    else:
        with pytest.raises(AssertionError):
            _assert_mtp3_graph_parity(result, log)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "data_parallel_size", "expected_backend"),
    [(8, 1, None), (1, 8, "deepep_auto")],
)
def test_deepseek_v4_dspark_rank_uses_standard_engine_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tensor_parallel_size: int,
    data_parallel_size: int,
    expected_backend: str | None,
) -> None:
    captured = {}
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=data_parallel_size,
        all2all_backend=expected_backend or "allgather_reducescatter",
        enable_expert_parallel=data_parallel_size > 1,
        world_size=tensor_parallel_size,
    )

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config)
            )

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(
        model_runtime,
        "_generate_with_llm",
        lambda *args, **kwargs: [{"token_ids": [1], "text": "ok"}],
    )
    monkeypatch.setattr(model_runtime, "_shutdown_llm", lambda llm: None)

    result = model_runtime._case_deepseek_v4_dspark_rank(
        DEEPSEEK_V4_MODEL,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=0.9,
    )

    assert captured["tokenizer_mode"] == "deepseek_v4"
    assert captured["kv_cache_dtype"] == "fp8"
    assert captured["speculative_config"] == (
        model_runtime.DEEPSEEK_V4_DSPARK_SPECULATIVE_CONFIG
    )
    assert captured["tensor_parallel_size"] == tensor_parallel_size
    assert captured["enable_expert_parallel"] is (data_parallel_size > 1)
    assert captured.get("all2all_backend") == expected_backend
    assert "prefill_context_parallel_size" not in captured
    assert "kv_transfer_config" not in captured
    assert "moe_backend" not in captured
    assert result["speculative_method"] == "dspark"
    assert result["draft_token_count"] == 7
    assert result["pcp_world_size"] == 1
    assert result["output"]


def test_deepseek_v4_dspark_cli_forwards_only_topology(monkeypatch, capsys) -> None:
    captured = {}

    def fake_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return {"output": []}

    monkeypatch.setattr(model_runtime, "_case_deepseek_v4_dspark", fake_case)

    assert model_runtime._main(
        [
            "deepseek-v4-dspark-smoke",
            "--model",
            str(DEEPSEEK_V4_MODEL),
            "--topology",
            "dp8_ep8",
            "--gpu-memory-utilization",
            "0.9",
        ]
    ) == 0
    assert captured == {
        "model_path": DEEPSEEK_V4_MODEL,
        "topology": "dp8_ep8",
        "gpu_memory_utilization": 0.9,
    }
    assert "VLLM_HCU_RESULT=" in capsys.readouterr().out


class _FakeEvent:
    def __init__(self):
        self._is_set = False

    def set(self):
        self._is_set = True

    def is_set(self):
        return self._is_set

    def wait(self, timeout=None):
        del timeout
        return self._is_set


class _FakeResultQueue:
    def __init__(self, error=None):
        self.error = error

    def get(self, timeout=None):
        del timeout
        if self.error is not None:
            raise self.error
        raise AssertionError("unexpected queue read")


class _FakeRankProcess:
    def __init__(
        self,
        pid,
        exitcode,
        args,
        join_error=None,
        start_hook=None,
        publish_pid_before_hook=True,
    ):
        self.pid = None
        self._started_pid = pid
        self.exitcode = exitcode
        self.args = args
        self.join_error = join_error
        self.start_hook = start_hook
        self.publish_pid_before_hook = publish_pid_before_hook
        self.child_created = False

    def start(self):
        self.child_created = True
        if self.publish_pid_before_hook:
            self.pid = self._started_pid
        for value in self.args:
            if isinstance(value, _FakeEvent):
                value.set()
        if self.start_hook is not None:
            self.start_hook()
        self.pid = self._started_pid

    def join(self, timeout=None):
        del timeout
        if self.join_error is not None:
            raise self.join_error

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.exitcode = -signal.SIGTERM

    def kill(self):
        self.exitcode = -signal.SIGKILL


class _FakeMultiprocessingContext:
    def __init__(
        self,
        exitcodes,
        *,
        join_error=None,
        queue_error=None,
        start_hook=None,
        publish_pid_before_hook=True,
    ):
        self.exitcodes = list(exitcodes)
        self.join_error = join_error
        self.queue = _FakeResultQueue(queue_error)
        self.processes = []
        self.start_hook = start_hook
        self.publish_pid_before_hook = publish_pid_before_hook

    def Event(self):
        return _FakeEvent()

    def Lock(self):
        return self

    def Value(self, typecode, value):
        del typecode
        return SimpleNamespace(value=value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def Queue(self):
        return self.queue

    def Process(self, *, target, args):
        del target
        index = len(self.processes)
        process = _FakeRankProcess(
            1000 + index,
            self.exitcodes[index],
            args,
            join_error=self.join_error,
            start_hook=self.start_hook,
            publish_pid_before_hook=self.publish_pid_before_hook,
        )
        self.processes.append(process)
        return process


def _run_fake_dp_case(monkeypatch, context):
    cleaned = []
    monkeypatch.setattr(
        model_runtime.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        model_runtime,
        "_terminate_data_parallel_process_groups",
        lambda processes, ready_events, process_group_id: cleaned.append(
            ([process.pid for process in processes], list(ready_events))
        ),
        raising=False,
    )
    kwargs = {
        "tensor_parallel_size": 1,
        "data_parallel_size": len(context.exitcodes),
        "gpu_memory_utilization": 0.9,
        "all2all_backend": "deepep_high_throughput",
        "moe_backend": "deep_gemm",
    }
    return cleaned, kwargs


def test_tp_ep_dp_uses_explicit_multiprocess_launcher(monkeypatch):
    captured = {}
    expected = {"parallel_config": {"data_parallel_size": 8}, "output": []}

    def fail_single_process_llm(**kwargs):
        raise AssertionError(f"single-process LLM was constructed: {kwargs}")

    def fake_data_parallel_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return expected

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fail_single_process_llm))
    monkeypatch.setattr(
        model_runtime,
        "_case_tp_ep_smoke_data_parallel",
        fake_data_parallel_case,
        raising=False,
    )

    result = model_runtime._case_tp_ep_smoke(
        Path("/models/fake"),
        tensor_parallel_size=1,
        data_parallel_size=8,
        gpu_memory_utilization=0.9,
        all2all_backend="deepep_high_throughput",
        moe_backend="deep_gemm",
    )

    assert result is expected
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "all2all_backend": "deepep_high_throughput",
        "moe_backend": "deep_gemm",
    }


def test_tp_ep_cli_forwards_data_parallel_and_all2all(monkeypatch, capsys):
    captured = {}

    def fake_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return {"output": []}

    monkeypatch.setattr(model_runtime, "_case_tp_ep_smoke", fake_case)
    assert model_runtime._main(
        [
            "tp-ep-smoke",
            "--model",
            "/models/fake",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "8",
            "--all2all-backend",
            "deepep_low_latency",
            "--moe-backend",
            "deep_gemm",
        ]
    ) == 0
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.6,
        "all2all_backend": "deepep_low_latency",
        "moe_backend": "deep_gemm",
    }
    assert "VLLM_HCU_RESULT=" in capsys.readouterr().out


def test_tp_ep_ll_exercises_model_specific_deepep_token_capacity(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(vllm_config=None)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(model_runtime, "_generate_with_llm", lambda *args, **kwargs: [])
    monkeypatch.setattr(model_runtime, "_shutdown_llm", lambda llm: None)

    model_runtime._case_tp_ep_smoke_rank(
        Path("/models/fake"),
        tensor_parallel_size=1,
        data_parallel_size=8,
        gpu_memory_utilization=0.9,
        all2all_backend="deepep_low_latency",
        moe_backend="deep_gemm",
    )

    assert captured["max_num_batched_tokens"] == 300


def test_tp_ep_default_does_not_override_vllm_all2all_backend_with_none(
    monkeypatch,
):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(vllm_config=None)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(model_runtime, "_generate_with_llm", lambda *args, **kwargs: [])
    monkeypatch.setattr(model_runtime, "_shutdown_llm", lambda llm: None)

    model_runtime._case_tp_ep_smoke_rank(
        Path("/models/fake"),
        tensor_parallel_size=4,
        data_parallel_size=1,
        gpu_memory_utilization=0.4,
        all2all_backend=None,
        moe_backend="aiter",
    )

    assert "all2all_backend" not in captured


def test_tp_ep_dp_cleans_every_rank_group_after_rank_failure(monkeypatch):
    context = _FakeMultiprocessingContext([1, None])
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(RuntimeError, match="rank process .* failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_tp_ep_dp_installs_sigterm_cleanup_handler(monkeypatch):
    context = _FakeMultiprocessingContext([1, None])
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)
    handlers = []
    active_handlers = {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }

    def fake_signal(sig, handler):
        handlers.append((sig, handler))
        previous = active_handlers[sig]
        active_handlers[sig] = handler
        return previous

    monkeypatch.setattr(
        model_runtime.signal,
        "signal",
        fake_signal,
    )

    with pytest.raises(RuntimeError, match="rank process .* failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert cleaned
    assert handlers[0][0] == signal.SIGTERM
    with pytest.raises(
        model_runtime._DataParallelTermination,
        match="received signal 15",
    ):
        handlers[0][1](signal.SIGTERM, None)
    assert active_handlers == {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }


def test_tp_ep_dp_tracks_rank_if_sigterm_interrupts_process_start(monkeypatch):
    active_handlers = {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }

    def fake_signal(sig, handler):
        previous = active_handlers[sig]
        active_handlers[sig] = handler
        return previous

    def interrupt_start():
        active_handlers[signal.SIGTERM](signal.SIGTERM, None)

    context = _FakeMultiprocessingContext([None], start_hook=interrupt_start)
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)
    monkeypatch.setattr(model_runtime.signal, "signal", fake_signal)

    with pytest.raises(
        model_runtime._DataParallelTermination,
        match="received signal 15",
    ):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000]]
    assert active_handlers == {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }


def test_tp_ep_dp_tracks_rank_if_keyboard_interrupts_process_start(monkeypatch):
    context = _FakeMultiprocessingContext(
        [None],
        start_hook=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(KeyboardInterrupt):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000]]


def test_tp_ep_dp_defers_sigint_until_child_pid_is_published(monkeypatch):
    active_handlers = {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }
    context = None

    def fake_signal(sig, handler):
        previous = active_handlers[sig]
        active_handlers[sig] = handler
        return previous

    def interrupt_after_child_creation():
        assert context is not None
        process = context.processes[0]
        assert process.child_created
        assert process.pid is None
        active_handlers[signal.SIGINT](signal.SIGINT, None)

    context = _FakeMultiprocessingContext(
        [None],
        start_hook=interrupt_after_child_creation,
        publish_pid_before_hook=False,
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)
    monkeypatch.setattr(model_runtime.signal, "signal", fake_signal)

    with pytest.raises(KeyboardInterrupt):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert context.processes[0].pid == 1000
    assert [entry[0] for entry in cleaned] == [[1000]]
    assert active_handlers == {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }


def test_tp_ep_dp_defers_second_sigterm_until_cleanup_finishes(monkeypatch):
    active_handlers = {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }
    cleanup_finished = []

    def fake_signal(sig, handler):
        previous = active_handlers[sig]
        active_handlers[sig] = handler
        return previous

    context = _FakeMultiprocessingContext([1])
    _, kwargs = _run_fake_dp_case(monkeypatch, context)
    monkeypatch.setattr(model_runtime.signal, "signal", fake_signal)

    def cleanup(processes, ready_events, process_group_id):
        del processes, ready_events, process_group_id
        active_handlers[signal.SIGTERM](signal.SIGTERM, None)
        cleanup_finished.append(True)

    monkeypatch.setattr(
        model_runtime,
        "_terminate_data_parallel_process_groups",
        cleanup,
    )

    with pytest.raises(
        model_runtime._DataParallelTermination,
        match="received signal 15",
    ):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert cleanup_finished == [True]
    assert active_handlers == {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }


def test_tp_ep_dp_defers_sigint_until_cleanup_finishes(monkeypatch):
    active_handlers = {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }
    cleanup_finished = []

    def fake_signal(sig, handler):
        previous = active_handlers[sig]
        active_handlers[sig] = handler
        return previous

    context = _FakeMultiprocessingContext([1])
    _, kwargs = _run_fake_dp_case(monkeypatch, context)
    monkeypatch.setattr(model_runtime.signal, "signal", fake_signal)

    def cleanup(processes, ready_events, process_group_id):
        del processes, ready_events, process_group_id
        active_handlers[signal.SIGINT](signal.SIGINT, None)
        cleanup_finished.append(True)

    monkeypatch.setattr(
        model_runtime,
        "_terminate_data_parallel_process_groups",
        cleanup,
    )

    with pytest.raises(KeyboardInterrupt):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert cleanup_finished == [True]
    assert active_handlers == {
        signal.SIGINT: signal.default_int_handler,
        signal.SIGTERM: signal.SIG_DFL,
    }


def test_tp_ep_dp_cleans_every_rank_group_after_interruption(monkeypatch):
    context = _FakeMultiprocessingContext(
        [None, None],
        join_error=KeyboardInterrupt(),
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(KeyboardInterrupt):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_tp_ep_dp_cleans_every_rank_group_after_queue_failure(monkeypatch):
    context = _FakeMultiprocessingContext(
        [0, 0],
        queue_error=RuntimeError("queue failed"),
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(RuntimeError, match="queue failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_owned_process_group_cleanup_escalates_and_verifies(monkeypatch):
    signals = []
    waits = iter([{1001}, set()])
    monkeypatch.setattr(
        model_runtime.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    monkeypatch.setattr(
        model_runtime,
        "_wait_for_process_groups",
        lambda pgids, timeout_s: next(waits),
        raising=False,
    )

    model_runtime._terminate_owned_process_groups(
        [1000, 1001],
        term_timeout_s=1,
        kill_timeout_s=1,
    )

    assert signals == [
        (1000, signal.SIGTERM),
        (1001, signal.SIGTERM),
        (1001, signal.SIGKILL),
    ]


def test_owned_process_group_cleanup_rejects_parent_group(monkeypatch):
    monkeypatch.setattr(model_runtime.os, "getpgrp", lambda: 2000)
    monkeypatch.setattr(model_runtime.os, "getppid", lambda: 2999)
    monkeypatch.setattr(model_runtime.os, "getpgid", lambda pid: 3000)

    with pytest.raises(RuntimeError, match="unvalidated process groups.*3000"):
        model_runtime._terminate_owned_process_groups([3000])


def test_owned_process_group_cleanup_reaps_leaders_while_waiting(monkeypatch):
    polled = []
    proc = SimpleNamespace(poll=lambda: polled.append(True))
    monkeypatch.setattr(model_runtime.os, "killpg", lambda pgid, sig: None)

    def fake_wait(pgids, timeout_s, *, process_leaders=()):
        del pgids, timeout_s
        for leader in process_leaders:
            leader.poll()
        return set()

    monkeypatch.setattr(model_runtime, "_wait_for_process_groups", fake_wait)

    model_runtime._terminate_owned_process_groups(
        [1000],
        process_leaders=[proc],
        term_timeout_s=1,
        kill_timeout_s=1,
    )

    assert polled == [True]


def test_case_group_cleanup_does_not_skip_an_exited_leader(monkeypatch):
    cleaned = []
    proc = SimpleNamespace(pid=2000, poll=lambda: 1, wait=lambda timeout: 1)
    monkeypatch.setattr(
        model_runtime,
        "_terminate_owned_process_groups",
        lambda pgids, **kwargs: cleaned.append(list(pgids)),
        raising=False,
    )

    model_runtime._terminate_case_process_group(proc)

    assert cleaned == [[2000]]


def test_dp_rank_joins_the_shared_owned_process_group(monkeypatch):
    calls = []
    installed_handlers = []
    process_group_id = SimpleNamespace(value=1234)
    ready = _FakeEvent()
    start_gate = _FakeEvent()
    start_gate.set()
    result_queue = SimpleNamespace(put=lambda result: calls.append(("put", result)))
    monkeypatch.setattr(
        model_runtime.os,
        "setpgid",
        lambda pid, pgid: calls.append((pid, pgid)),
    )
    monkeypatch.setattr(
        model_runtime.signal,
        "signal",
        lambda sig, handler: installed_handlers.append((sig, handler)),
    )
    monkeypatch.setattr(
        model_runtime,
        "_case_tp_ep_smoke_rank",
        lambda *args, **kwargs: {"output": []},
    )

    model_runtime._tp_ep_data_parallel_rank(
        1,
        2,
        "127.0.0.1",
        12345,
        Path("/models/fake"),
        1,
        0.9,
        "deepep_high_throughput",
        "deep_gemm",
        result_queue,
        process_group_id,
        _FakeMultiprocessingContext([]),
        ready,
        start_gate,
    )

    assert calls[0] == (0, 1234)
    assert installed_handlers == [
        (signal.SIGTERM, signal.SIG_DFL),
        (signal.SIGINT, signal.default_int_handler),
    ]
    assert ready.is_set()
