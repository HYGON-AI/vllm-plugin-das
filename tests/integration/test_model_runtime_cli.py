# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-safe CLI coverage for the model runtime subprocess harness."""

from pathlib import Path
import sys
from types import SimpleNamespace

from tests.integration import model_runtime


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
        moe_backend="dpsk_deep_gemm",
    )

    assert result is expected
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "all2all_backend": "deepep_high_throughput",
        "moe_backend": "dpsk_deep_gemm",
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
            "dpsk_deep_gemm",
        ]
    ) == 0
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.6,
        "all2all_backend": "deepep_low_latency",
        "moe_backend": "dpsk_deep_gemm",
    }
    assert "VLLM_HCU_RESULT=" in capsys.readouterr().out


def test_tp_ep_ll_uses_supported_deepep_token_capacity(monkeypatch):
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
        moe_backend="dpsk_deep_gemm",
    )

    assert captured["max_num_batched_tokens"] == 256
