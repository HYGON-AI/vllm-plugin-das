# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable tests for EvalScope report pass criteria."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import psutil

from tests.integration.server import evalscope_server
from tests.integration.server import evalscope_secure_cli
from tests.integration.server.evalscope_server import (
    _assert_pass_criteria,
    _direct_urlopen,
    _evaluation_environment,
    _server_environment,
    evalscope_command,
)


class _HealthyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def test_direct_urlopen_bypasses_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthyHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    try:
        with _direct_urlopen(
            f"http://127.0.0.1:{server.server_port}/health", timeout=1
        ) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_server_environment_clears_all_proxy_spellings_and_sets_local_no_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "example.internal")
    monkeypatch.setenv("no_proxy", "legacy.internal")

    environment = _server_environment()

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        assert name not in environment
    assert environment["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert environment["no_proxy"] == "127.0.0.1,localhost,::1"


def test_server_environment_honors_explicit_proxy_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://inherited.invalid:8080")

    environment = _server_environment(
        {"server": {"environment": {"HTTPS_PROXY": "http://explicit:8080"}}}
    )

    assert environment["HTTPS_PROXY"] == "http://explicit:8080"
    assert "127.0.0.1" in environment["NO_PROXY"]


def test_evaluation_environment_preserves_proxy_and_removes_host_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("GH_TOKEN", "host-secret")
    monkeypatch.setenv("GITHUB_PAT", "host-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "host-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/tmp/aws-credentials")
    monkeypatch.setenv(
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/secret"
    )
    monkeypatch.setenv(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://169.254.170.2/secret"
    )
    monkeypatch.setenv(
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "/tmp/aws-auth-token"
    )
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", '{"auths":{"registry":{}}}')
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/git-askpass")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.extraHeader")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Authorization: host-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:secret@packages.invalid/simple")
    monkeypatch.setenv(
        "PIP_EXTRA_INDEX_URL", "https://user:secret@extra.invalid/simple"
    )
    monkeypatch.setenv("UV_INDEX_URL", "https://user:secret@packages.invalid/simple")
    monkeypatch.setenv(
        "UV_EXTRA_INDEX_URL", "https://user:secret@extra.invalid/simple"
    )
    monkeypatch.setenv("openai_api_key", "host-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("KUBECONFIG", "/tmp/kubeconfig")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    original_home = os.environ.get("HOME")
    config = {
        "server": {"environment": {"VLLM_USE_V2_MODEL_RUNNER": "1"}},
        "evalscope": {"api_key": "eval-secret"},
    }

    environment = _evaluation_environment(config)

    assert environment["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert environment["VLLM_HCU_EVALSCOPE_API_KEY"] == "eval-secret"
    assert "GH_TOKEN" not in environment
    assert "GITHUB_PAT" not in environment
    assert "AWS_ACCESS_KEY_ID" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "AWS_SHARED_CREDENTIALS_FILE" not in environment
    assert "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" not in environment
    assert "AWS_CONTAINER_CREDENTIALS_FULL_URI" not in environment
    assert "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE" not in environment
    assert "DOCKER_AUTH_CONFIG" not in environment
    assert "GIT_ASKPASS" not in environment
    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_EXTRA_INDEX_URL" not in environment
    assert "UV_INDEX_URL" not in environment
    assert "UV_EXTRA_INDEX_URL" not in environment
    assert "openai_api_key" not in environment
    assert "SSH_AUTH_SOCK" not in environment
    assert "KUBECONFIG" not in environment
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert environment["HOME"] != original_home
    assert "127.0.0.1" in environment["NO_PROXY"]


def test_evalscope_command_keeps_api_key_out_of_argv(tmp_path: Path) -> None:
    secret = "must-not-appear-in-process-list"
    config = {
        "model": "/models/Qwen3-8B",
        "server": {"served_model_name": "Qwen3-8B"},
        "evalscope": {
            "api_key": secret,
            "eval_type": "openai_api",
            "generation_config": {"temperature": 0},
            "dataset_args": {"humaneval": {}},
            "eval_batch_size": 8,
            "timeout": 60,
            "limit": 16,
            "datasets": ["humaneval"],
        },
    }

    command = evalscope_command(
        config,
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        host="127.0.0.1",
        port=8000,
        work_dir=tmp_path,
    )

    assert secret not in command
    assert "--api-key" not in command
    assert command[:3] == [
        sys.executable,
        "-m",
        "tests.integration.server.evalscope_secure_cli",
    ]


def test_secure_cli_python_argv_does_not_change_proc_cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = f"evalscope-secret-{uuid.uuid4()}"
    monkeypatch.setenv("VLLM_HCU_EVALSCOPE_API_KEY", marker)
    monkeypatch.setattr(sys, "argv", ["evalscope-secure", "eval", "--help"])

    def assert_proc_cmdline_is_clean() -> int:
        assert marker.encode() not in Path("/proc/self/cmdline").read_bytes()
        assert sys.argv[-2:] == ["--api-key", marker]
        return 0

    monkeypatch.setattr(evalscope_secure_cli, "run_cmd", assert_proc_cmdline_is_clean)

    assert evalscope_secure_cli.main() == 0
    assert "VLLM_HCU_EVALSCOPE_API_KEY" not in os.environ


def test_humaneval_rejects_unsandboxed_host_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_HUMANEVAL_ISOLATED", raising=False)
    config = {"evalscope": {"datasets": ["humaneval"]}}

    with pytest.raises(RuntimeError, match="EvalScope sandbox.*isolated container"):
        evalscope_server._require_humaneval_execution_isolation(config)


def test_humaneval_allows_explicit_evalscope_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_HUMANEVAL_ISOLATED", raising=False)
    config = {
        "evalscope": {
            "datasets": ["humaneval"],
            "sandbox": {"enabled": True, "engine": "docker"},
        }
    }

    evalscope_server._require_humaneval_execution_isolation(config)


def test_humaneval_allows_explicit_isolated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_HCU_HUMANEVAL_ISOLATED", "1")

    evalscope_server._require_humaneval_execution_isolation(
        {"evalscope": {"datasets": ["humaneval"]}}
    )


class _ProbeResponse:
    def __init__(self, body: str, *, status: int = 200) -> None:
        self.status = status
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _prefix_probe_config() -> dict:
    return {
        "model": "/models/Qwen3-8B",
        "server": {
            "served_model_name": "qwen3-8b-gfx938",
            "prefix_probe": {
                "metric": "vllm:prefix_cache_hits_total",
                "timeout_s": 30,
                "max_tokens": 32,
                "content_repeat": 4,
                "messages": [
                    {
                        "role": "user",
                        "content": "shared prefix " * 256 + "return OK",
                    }
                ],
            },
        },
    }


def test_prefix_probe_sends_identical_requests_and_records_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict | None]] = []
    metrics = iter(
        (
            "vllm:prefix_cache_hits_total{engine=\"0\"} 2\n",
            "vllm:prefix_cache_hits_total{engine=\"0\"} 9\n",
        )
    )

    def direct_open(target, *, timeout: int):
        assert timeout == 30
        if isinstance(target, str):
            calls.append((target, None))
            return _ProbeResponse(next(metrics))
        payload = json.loads(target.data.decode())
        calls.append((target.full_url, payload))
        return _ProbeResponse(
            json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "OK"}}
                    ]
                }
            )
        )

    monkeypatch.setattr(evalscope_server, "_direct_urlopen", direct_open)
    run_probe = getattr(evalscope_server, "_run_prefix_probe", None)
    assert callable(run_probe), "prefix probe runner is not implemented"

    before, after, metrics_path = run_probe(
        _prefix_probe_config(),
        host="127.0.0.1",
        port=10128,
        work_dir=tmp_path,
    )

    assert (before, after) == (2.0, 9.0)
    assert metrics_path == tmp_path / "logs/metrics.prom"
    assert metrics_path.read_text(encoding="utf-8").endswith('} 9\n')
    request_payloads = [payload for _, payload in calls if payload is not None]
    assert len(request_payloads) == 2
    assert request_payloads[0] == request_payloads[1]
    assert request_payloads[0]["model"] == "qwen3-8b-gfx938"
    assert request_payloads[0]["messages"][0]["content"] == (
        ("shared prefix " * 256 + "return OK") * 4
    )
    assert [url for url, _ in calls] == [
        "http://127.0.0.1:10128/metrics",
        "http://127.0.0.1:10128/v1/chat/completions",
        "http://127.0.0.1:10128/v1/chat/completions",
        "http://127.0.0.1:10128/metrics",
    ]


def test_prefix_probe_requires_metric_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def direct_open(target, *, timeout: int):
        del timeout
        if isinstance(target, str):
            return _ProbeResponse("vllm:prefix_cache_hits_total 4\n")
        return _ProbeResponse(
            '{"choices":[{"message":{"role":"assistant","content":"OK"}}]}'
        )

    monkeypatch.setattr(evalscope_server, "_direct_urlopen", direct_open)
    run_probe = getattr(evalscope_server, "_run_prefix_probe", None)
    assert callable(run_probe), "prefix probe runner is not implemented"

    with pytest.raises(AssertionError, match="did not increase"):
        run_probe(
            _prefix_probe_config(),
            host="127.0.0.1",
            port=10128,
            work_dir=tmp_path,
        )


def test_prefix_probe_accepts_reasoning_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = iter(
        (
            "vllm:prefix_cache_hits_total 0\n",
            "vllm:prefix_cache_hits_total 8\n",
        )
    )

    def direct_open(target, *, timeout: int):
        del timeout
        if isinstance(target, str):
            return _ProbeResponse(next(metrics))
        return _ProbeResponse(
            '{"choices":[{"message":{"role":"assistant","content":null,'
            '"reasoning_content":"working"}}]}'
        )

    monkeypatch.setattr(evalscope_server, "_direct_urlopen", direct_open)

    before, after, _ = evalscope_server._run_prefix_probe(
        _prefix_probe_config(),
        host="127.0.0.1",
        port=10128,
        work_dir=tmp_path,
    )

    assert (before, after) == (0.0, 8.0)


def test_prefix_probe_is_optional_without_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evalscope_server,
        "_direct_urlopen",
        lambda *_args, **_kwargs: pytest.fail("optional probe used the network"),
    )

    assert evalscope_server._run_prefix_probe(
        {"model": "/models/Qwen3-8B", "server": {}},
        host="127.0.0.1",
        port=10128,
        work_dir=tmp_path,
    ) is None


def test_server_environment_does_not_force_flash_attention_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", raising=False)

    environment = _server_environment()

    assert "VLLM_HCU_USE_FLASH_ATTN_UNIFIED" not in environment


def test_reset_evalscope_artifacts_removes_stale_outputs_only(
    tmp_path: Path,
) -> None:
    (tmp_path / evalscope_server.EVALSCOPE_OWNER_MARKER).write_text(
        evalscope_server.EVALSCOPE_OWNER_SIGNATURE,
        encoding="utf-8",
    )
    for name in ("configs", "predictions", "reviews", "reports"):
        artifact = tmp_path / name / "stale.jsonl"
        artifact.parent.mkdir()
        artifact.write_text("stale", encoding="utf-8")
    log = tmp_path / "logs/evalscope.log"
    log.parent.mkdir()
    log.write_text("keep", encoding="utf-8")

    evalscope_server._reset_evalscope_artifacts(tmp_path)

    assert not any(
        (tmp_path / name).exists()
        for name in ("configs", "predictions", "reviews", "reports")
    )
    assert log.read_text(encoding="utf-8") == "keep"


def test_reset_evalscope_artifacts_rejects_unowned_broad_path(
    tmp_path: Path,
) -> None:
    broad = tmp_path / "models"
    report = broad / "reports/keep.json"
    report.parent.mkdir(parents=True)
    report.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership marker"):
        evalscope_server._reset_evalscope_artifacts(broad)

    assert report.read_text(encoding="utf-8") == "keep"


def test_reset_evalscope_artifacts_rejects_symlinked_work_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    report = target / "reports/keep.json"
    report.parent.mkdir(parents=True)
    report.write_text("keep", encoding="utf-8")
    work_dir = tmp_path / "work-dir"
    work_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        evalscope_server._reset_evalscope_artifacts(work_dir)

    assert report.read_text(encoding="utf-8") == "keep"


def test_terminate_process_group_also_stops_new_session_descendants(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import time; time.sleep(300)'],start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(300)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(child_pid_path)],
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not child_pid_path.exists():
            time.sleep(0.05)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        evalscope_server._terminate_process_group(parent, timeout_s=2)

        assert parent.poll() is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and psutil.pid_exists(child_pid):
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid) or (
            psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
        if child_pid is not None and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()


def test_terminate_process_group_stops_owned_orphan_after_parent_exit(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "orphan.pid"
    owner_token = f"evalscope-test-{os.getpid()}-{time.monotonic_ns()}"
    parent_code = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import time; time.sleep(300)'],start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    env = os.environ.copy()
    env["VLLM_HCU_EVAL_PROCESS_OWNER"] = owner_token
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(child_pid_path)],
        env=env,
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        parent.wait(timeout=10)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert psutil.pid_exists(child_pid)

        evalscope_server._terminate_process_group(
            parent,
            timeout_s=2,
            owner_token=owner_token,
        )

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and psutil.pid_exists(child_pid):
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid) or (
            psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if child_pid is not None and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()


def test_terminate_process_group_signals_detached_children_concurrently(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_ready_path = tmp_path / "child.ready"
    child_term_path = tmp_path / "child.term"
    parent_ready_path = tmp_path / "parent.ready"
    parent_term_path = tmp_path / "parent.term"
    owner_token = f"evalscope-test-{os.getpid()}-{time.monotonic_ns()}"
    child_code = (
        "import pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM,lambda *_:"
        "pathlib.Path(sys.argv[1]).write_text(str(time.monotonic())));"
        "pathlib.Path(sys.argv[2]).write_text('ready');"
        "time.sleep(300)"
    )
    parent_code = (
        "import pathlib,signal,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1],"
        "sys.argv[2],sys.argv[3]],start_new_session=True);"
        "child_ready=pathlib.Path(sys.argv[3]);"
        "exec('while not child_ready.exists():\\n time.sleep(0.01)');"
        "pathlib.Path(sys.argv[4]).write_text(str(child.pid));"
        "signal.signal(signal.SIGTERM,lambda *_:"
        "pathlib.Path(sys.argv[5]).write_text(str(time.monotonic())));"
        "pathlib.Path(sys.argv[6]).write_text('ready');"
        "time.sleep(300)"
    )
    env = os.environ.copy()
    env["VLLM_HCU_EVAL_PROCESS_OWNER"] = owner_token
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            child_code,
            str(child_term_path),
            str(child_ready_path),
            str(child_pid_path),
            str(parent_term_path),
            str(parent_ready_path),
        ],
        env=env,
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not parent_ready_path.exists():
            time.sleep(0.05)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        evalscope_server._terminate_process_group(
            parent,
            timeout_s=1,
            owner_token=owner_token,
        )

        parent_term = float(parent_term_path.read_text(encoding="utf-8"))
        child_term = float(child_term_path.read_text(encoding="utf-8"))
        assert abs(parent_term - child_term) < 0.5
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
        if child_pid is not None and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()


def _config(score: float) -> dict:
    return {
        "model": "/models/Qwen3-8B",
        "evalscope": {
            "pass_criteria": {
                "dataset": "gsm8k",
                "metric": "mean_acc",
                "display_name": "Pass@1",
                "minimum_score": score,
            }
        },
    }


def _write_report(work_dir: Path, score: float) -> Path:
    report_path = work_dir / "reports/Qwen3-8B/gsm8k.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "name": "Qwen3-8B@gsm8k",
                "dataset_name": "gsm8k",
                "model_name": "Qwen3-8B",
                "metrics": [
                    {
                        "name": "mean_acc",
                        "score": score,
                        "num": 100,
                        "categories": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _write_schema_v2_report(work_dir: Path, score: float) -> Path:
    report_path = work_dir / "reports/Qwen3-8B/gsm8k.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_name": "gsm8k",
                "model_name": "Qwen3-8B",
                "metrics": [
                    {
                        "identity": {
                            "name": "accuracy",
                            "aggregation": "mean",
                            "dimensions": {},
                        },
                        "score": score,
                        "num": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_pass_at_one_accepts_score_at_threshold(tmp_path: Path) -> None:
    _write_report(tmp_path, 0.95)

    _assert_pass_criteria(
        _config(0.95),
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )

    assert "Pass@1=0.9500" in (tmp_path / "logs/evalscope.log").read_text()


def test_pass_at_one_accepts_evalscope_schema_v2_identity(tmp_path: Path) -> None:
    _write_schema_v2_report(tmp_path, 0.95)

    _assert_pass_criteria(
        _config(0.95),
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )

    assert "Pass@1=0.9500" in (tmp_path / "logs/evalscope.log").read_text()


def test_pass_at_one_rejects_score_below_threshold(tmp_path: Path) -> None:
    _write_report(tmp_path, 0.9499)

    with pytest.raises(AssertionError, match=r"Pass@1=0\.9499"):
        _assert_pass_criteria(
            _config(0.95),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )


def test_pass_at_one_requires_report_metric(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="missing EvalScope report metric"):
        _assert_pass_criteria(
            _config(0.95),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )


def _exact_humaneval_config() -> dict:
    return {
        "model": "/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
        "server": {"served_model_name": "DeepSeek-V4-Flash-0731-Channel-FP8-w8a8"},
        "evalscope": {
            "pass_criteria": {
                "dataset": "humaneval",
                "num_predictions": 32,
                "num_reviews": 32,
                "normalize_code_fences": True,
                "mean_acc": 1.0,
                "mean_acc_pass@1": 1.0,
            }
        },
    }


def _write_exact_humaneval_artifacts(
    work_dir: Path,
    *,
    score: float = 0.90625,
    records: int = 32,
    model: str = "DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
) -> None:
    report_path = work_dir / f"reports/{model}/humaneval.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "humaneval",
                "model_name": model,
                "metrics": [
                    {"name": "mean_acc", "score": score, "num": 32},
                    {
                        "name": "mean_acc_pass@1",
                        "score": score,
                        "num": 32,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction_path = work_dir / f"predictions/{model}/humaneval.jsonl"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text(
        "".join(
            json.dumps({"index": index}) + "\n" for index in range(records)
        ),
        encoding="utf-8",
    )
    review_path = work_dir / f"reviews/{model}/humaneval.jsonl"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(
        "".join(
            json.dumps(
                {
                    "index": index,
                    "sample_score": {
                        "score": {
                            "prediction": (
                                "```python\ndef candidate(value):\n"
                                "    return value"
                            )
                        },
                        "sample_metadata": {
                            "task_id": f"HumanEval/{index}",
                            "entry_point": "candidate",
                            "prompt": "def candidate(value):\n",
                            "test": "def check(candidate):\n"
                            "    assert candidate(1) == 1\n",
                        },
                    },
                }
            )
            + "\n"
            for index in range(records)
        ),
        encoding="utf-8",
    )


def _accept_normalized_humaneval(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    completions: list[str] = []

    def check(problem: dict, completion: str, timeout: int) -> dict:
        del problem, timeout
        completions.append(completion)
        return {"passed": completion == "def candidate(value):\n    return value"}

    monkeypatch.setattr(evalscope_server, "_check_humaneval_completion", check)
    return completions


def test_humaneval_fence_normalization_preserves_body_indentation() -> None:
    assert evalscope_server._normalize_humaneval_completion(
        "```python\n    return value\n```"
    ) == "    return value"


def test_exact_humaneval_criteria_accepts_both_metrics_and_artifact_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _accept_normalized_humaneval(monkeypatch)
    _write_exact_humaneval_artifacts(tmp_path)

    _assert_pass_criteria(
        _exact_humaneval_config(),
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )

    assert completions == ["def candidate(value):\n    return value"] * 32
    log = (tmp_path / "logs/evalscope.log").read_text()
    assert "raw_mean_acc=0.9062" in log
    assert "normalized_mean_acc=1.0000" in log
    assert "raw_mean_acc_pass@1=0.9062" in log
    assert "normalized_mean_acc_pass@1=1.0000" in log
    assert "predictions=32" in log
    assert "reviews=32" in log
    normalized_report = json.loads(
        (tmp_path / "reports/normalized_humaneval.json").read_text()
    )
    assert normalized_report["passed"] == 32
    assert normalized_report["score"] == 1.0


def test_exact_humaneval_accepts_evalscope_sanitized_model_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_normalized_humaneval(monkeypatch)
    _write_exact_humaneval_artifacts(
        tmp_path,
        model="GLM-5_1-Channel-FP8-w8a8",
    )
    config = _exact_humaneval_config()
    config["model"] = "/models/GLM-5___1-Channel-FP8-w8a8"
    config["server"]["served_model_name"] = "GLM-5___1-Channel-FP8-w8a8"

    _assert_pass_criteria(
        config,
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )


def test_exact_humaneval_criteria_rejects_partial_artifacts(tmp_path: Path) -> None:
    _write_exact_humaneval_artifacts(tmp_path, records=31)

    with pytest.raises(AssertionError, match="expected 32 predictions, got 31"):
        _assert_pass_criteria(
            _exact_humaneval_config(),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )


def test_exact_humaneval_criteria_rejects_normalized_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_exact_humaneval_artifacts(tmp_path)

    def reject_last(problem: dict, completion: str, timeout: int) -> dict:
        del completion, timeout
        return {"passed": problem["task_id"] != "HumanEval/31"}

    monkeypatch.setattr(
        evalscope_server,
        "_check_humaneval_completion",
        reject_last,
    )

    with pytest.raises(
        AssertionError,
        match=r"normalized HumanEval expected 32 passed, got 31.*HumanEval/31",
    ):
        _assert_pass_criteria(
            _exact_humaneval_config(),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )


def test_exact_humaneval_diagnostic_profile_records_score_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_exact_humaneval_artifacts(tmp_path)
    config = _exact_humaneval_config()
    config["evalscope"]["pass_criteria"]["enforce_score"] = False

    monkeypatch.setattr(
        evalscope_server,
        "_check_humaneval_completion",
        lambda *_args, **_kwargs: pytest.fail(
            "diagnostic profiles must not execute generated code twice"
        ),
    )

    _assert_pass_criteria(
        config,
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )

    assert "diagnostic criterion" in (
        tmp_path / "logs/evalscope.log"
    ).read_text()


def test_evalscope_sandbox_does_not_reexecute_completion_on_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_exact_humaneval_artifacts(tmp_path, score=1.0)
    config = _exact_humaneval_config()
    config["evalscope"]["sandbox"] = {
        "enabled": True,
        "engine": "docker",
    }
    monkeypatch.setattr(
        evalscope_server,
        "_check_humaneval_completion",
        lambda *_args, **_kwargs: pytest.fail(
            "sandboxed EvalScope output must not be reexecuted on the host"
        ),
    )

    _assert_pass_criteria(
        config,
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )
