# SPDX-License-Identifier: Apache-2.0
"""Protocol-server GPU-memory waits must precede model startup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.integration.gpu_memory import GpuMemoryUnavailableError
from tests.integration.server import openai_server


@pytest.mark.parametrize(
    ("extra_args", "cards", "fraction"),
    [
        (None, 1, 0.2),
        (["--tensor-parallel-size", "2", "--gpu-memory-utilization=0.35"], 2, 0.35),
    ],
)
def test_protocol_wait_precedes_launch_and_preserves_startup_budget(
    tmp_path, monkeypatch, extra_args, cards, fraction,
):
    events = []
    observed = {}
    clock = {"seconds": 0}
    process = SimpleNamespace(pid=12345)
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("VLLM_HCU_INTEGRATION_LOG_DIR", str(log_dir))
    monkeypatch.setenv("VLLM_HCU_PROTOCOL_GPU_MEMORY_UTILIZATION", "0.2")
    monkeypatch.setattr(openai_server, "_available_port", lambda: 18555)

    def wait(**kwargs):
        events.append("wait")
        observed["wait"] = kwargs
        assert kwargs["required_cards"] == cards
        assert kwargs["gpu_memory_utilization"] == fraction
        kwargs["report"]("GPU memory ready")
        clock["seconds"] += 600

    def launch(command, **kwargs):
        events.append("launch")
        observed["launch_env"] = kwargs["env"]
        assert clock["seconds"] == 600
        return process

    def health(proc, url, timeout_s, log_path):
        events.append("health")
        assert proc is process
        assert timeout_s == 37
        assert clock["seconds"] == 600

    def cleanup(proc):
        assert proc is process
        events.append("cleanup")

    monkeypatch.setattr(openai_server, "wait_for_gpu_memory", wait)
    monkeypatch.setattr(openai_server.subprocess, "Popen", launch)
    monkeypatch.setattr(openai_server, "_wait_for_health", health)
    monkeypatch.setattr(openai_server, "_terminate_process_group", cleanup)
    kwargs = {"startup_timeout_s": 37, "extra_args": extra_args}
    if cards != 1:
        kwargs["required_hcu_count"] = cards
    with openai_server.serve_openai_protocol_model(tmp_path / "model", **kwargs):
        events.append("yield")

    assert events == ["wait", "launch", "health", "yield", "cleanup"]
    assert observed["wait"]["env"] is observed["launch_env"]
    assert observed["launch_env"]["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] == "1"
    assert "GPU memory ready" in next(log_dir.glob("*.log")).read_text()


def test_protocol_memory_wait_timeout_does_not_launch_server(tmp_path, monkeypatch):
    events = []
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("VLLM_HCU_INTEGRATION_LOG_DIR", str(log_dir))
    monkeypatch.setattr(openai_server, "_available_port", lambda: 18555)

    def timeout(**kwargs):
        events.append("wait")
        kwargs["report"]("GPU memory unavailable")
        raise GpuMemoryUnavailableError("GPU busy after 600s")

    def unexpected_launch(*args, **kwargs):
        pytest.fail("server must not launch after the memory deadline")

    monkeypatch.setattr(openai_server, "wait_for_gpu_memory", timeout)
    monkeypatch.setattr(openai_server.subprocess, "Popen", unexpected_launch)
    monkeypatch.setattr(openai_server, "_wait_for_health", unexpected_launch)
    with pytest.raises(GpuMemoryUnavailableError, match="GPU busy after 600s"):
        with openai_server.serve_openai_protocol_model(tmp_path / "model"):
            pytest.fail("unavailable device must not yield a server")

    assert events == ["wait"]
    assert "GPU memory unavailable" in next(log_dir.glob("*.log")).read_text()
