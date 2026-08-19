# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared OpenAI server + EvalScope integration-test runner."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]


def load_config(default_path: Path, config_env: str) -> dict[str, Any]:
    config_path = Path(os.environ.get(config_env, str(default_path)))
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"invalid eval config: {config_path}")
    return config


def _available_hcu_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _model_path(config: dict[str, Any], model_env: str) -> str:
    return os.environ.get(model_env, str(config["model"]))


def _require_runtime(
    config: dict[str, Any],
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> None:
    model = Path(_model_path(config, model_env))
    if not model.exists():
        pytest.skip(f"{model_label} model path is unavailable: {model}")
    if shutil.which("vllm") is None:
        pytest.skip("vllm CLI is unavailable")
    if shutil.which("evalscope") is None:
        pytest.skip("evalscope CLI is unavailable")
    hcu_count = _available_hcu_count()
    if hcu_count < required_hcu_count:
        pytest.skip(
            f"{model_label} test requires {required_hcu_count} HCU devices, "
            f"got {hcu_count}"
        )


def _maybe_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def server_command(
    config: dict[str, Any], *, model_env: str
) -> tuple[list[str], str, int]:
    server = config["server"]
    model = _model_path(config, model_env)
    host = os.environ.get("VLLM_HCU_SERVER_HOST", str(server.get("host", "127.0.0.1")))
    port = _maybe_int_env("VLLM_HCU_SERVER_PORT", int(server.get("port", 10128)))
    args = [str(item) for item in server.get("args", [])]
    if "--port" not in args and "-p" not in args:
        args.extend(["--port", str(port)])
    return ["vllm", "serve", model, *args], host, port


def evalscope_command(
    config: dict[str, Any],
    *,
    model_env: str,
    host: str,
    port: int,
    work_dir: Path,
) -> list[str]:
    evalscope = config["evalscope"]
    model = _model_path(config, model_env)
    generation = evalscope["generation_config"]
    dataset_args = json.dumps(evalscope["dataset_args"], separators=(",", ":"))
    command = [
        "evalscope",
        "eval",
        "--model",
        model,
        "--api-url",
        f"http://{host}:{port}/v1",
        "--api-key",
        str(evalscope.get("api_key", "EMPTY")),
        "--eval-type",
        str(evalscope.get("eval_type", "openai_api")),
        "--generation-config",
        json.dumps(generation, separators=(",", ":")),
    ]
    if evalscope.get("stream", False):
        command.append("--stream")
    command.extend(
        [
            "--eval-batch-size",
            str(os.environ.get("VLLM_HCU_EVAL_BATCH_SIZE", evalscope["eval_batch_size"])),
            "--timeout",
            str(evalscope["timeout"]),
            "--limit",
            str(os.environ.get("VLLM_HCU_EVAL_LIMIT", evalscope["limit"])),
            "--datasets",
            ",".join(str(item) for item in evalscope["datasets"]),
            "--dataset-args",
            dataset_args,
            "--work-dir",
            str(work_dir),
        ]
    )
    if evalscope.get("no_timestamp", True):
        command.append("--no-timestamp")
    return command


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab")


def _wait_for_server(proc: subprocess.Popen, url: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM server exited before ready, rc={returncode}")
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def _terminate_process_group(proc: subprocess.Popen, timeout_s: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout_s)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=10)


def _server_environment(config: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if config is not None:
        configured = config.get("server", {}).get("environment", {})
        if not isinstance(configured, dict):
            raise TypeError("server.environment must be a mapping")
        env.update({str(name): str(value) for name, value in configured.items()})
    return env


def _report_metric(
    work_dir: Path,
    *,
    model: str,
    dataset: str,
    metric: str,
) -> tuple[float, int, Path]:
    reports_dir = work_dir / "reports"
    report_paths = sorted(
        reports_dir.rglob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    expected_model_names = {model, Path(model).name}
    available: list[str] = []
    for report_path in report_paths:
        try:
            with report_path.open(encoding="utf-8") as stream:
                report = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        report_dataset = str(report.get("dataset_name", ""))
        report_model = str(report.get("model_name", ""))
        metrics = report.get("metrics")
        if not isinstance(metrics, list):
            continue
        for report_metric in metrics:
            if not isinstance(report_metric, dict):
                continue
            metric_name = str(report_metric.get("name", ""))
            available.append(f"{report_model}/{report_dataset}/{metric_name}")
            if (
                report_dataset.casefold() != dataset.casefold()
                or report_model not in expected_model_names
                or metric_name.casefold() != metric.casefold()
            ):
                continue
            score = report_metric.get("score")
            num = report_metric.get("num")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise AssertionError(
                    f"invalid {dataset} {metric} score in {report_path}: {score!r}"
                )
            if not isinstance(num, int) or isinstance(num, bool):
                raise AssertionError(
                    f"invalid {dataset} {metric} sample count in "
                    f"{report_path}: {num!r}"
                )
            return float(score), num, report_path
    rendered = ", ".join(available) if available else "none"
    raise AssertionError(
        f"missing EvalScope report metric model={Path(model).name!r}, "
        f"dataset={dataset!r}, metric={metric!r} under {reports_dir}; "
        f"available={rendered}"
    )


def _assert_pass_criteria(
    config: dict[str, Any],
    *,
    model_env: str,
    work_dir: Path,
    eval_log_path: Path,
) -> None:
    criteria = config["evalscope"].get("pass_criteria")
    if criteria is None:
        return
    if not isinstance(criteria, dict):
        raise TypeError("evalscope.pass_criteria must be a mapping")
    dataset = str(criteria["dataset"])
    metric = str(criteria["metric"])
    display_name = str(criteria.get("display_name", metric))
    minimum_score = float(criteria["minimum_score"])
    model = _model_path(config, model_env)
    score, num, report_path = _report_metric(
        work_dir,
        model=model,
        dataset=dataset,
        metric=metric,
    )
    verdict = (
        f"pass criterion: {dataset} {display_name}={score:.4f}, "
        f"required>={minimum_score:.4f}, samples={num}, report={report_path}"
    )
    with _open_log(eval_log_path) as eval_log:
        eval_log.write((verdict + "\n").encode())
    assert score >= minimum_score, verdict


def run_evalscope_server_test(
    config: dict[str, Any],
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> None:
    _require_runtime(
        config,
        model_env=model_env,
        model_label=model_label,
        required_hcu_count=required_hcu_count,
    )
    command, host, port = server_command(config, model_env=model_env)
    startup_timeout = _maybe_int_env(
        "VLLM_HCU_SERVER_STARTUP_TIMEOUT",
        int(config["server"].get("startup_timeout_s", 3600)),
    )
    shutdown_timeout = int(config["server"].get("shutdown_timeout_s", 60))
    work_dir = Path(
        os.environ.get("VLLM_HCU_EVAL_WORK_DIR", config["evalscope"]["work_dir"])
    )
    log_dir = work_dir / "logs"
    server_log_path = log_dir / "vllm_server.log"
    eval_log_path = log_dir / "evalscope.log"

    env = _server_environment(config)

    with _open_log(server_log_path) as server_log:
        server_log.write(("server command: " + " ".join(command) + "\n").encode())
        server_log.write(b"server environment: VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1\n")
        server_log.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(proc, f"http://{host}:{port}/health", startup_timeout)
            eval_command = evalscope_command(
                config,
                model_env=model_env,
                host=host,
                port=port,
                work_dir=work_dir,
            )
            with _open_log(eval_log_path) as eval_log:
                eval_log.write(("eval command: " + " ".join(eval_command) + "\n").encode())
                eval_log.flush()
                result = subprocess.run(
                    eval_command,
                    cwd=ROOT,
                    env=env,
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            assert result.returncode == 0, (
                f"evalscope failed with rc={result.returncode}; "
                f"server_log={server_log_path}; eval_log={eval_log_path}"
            )
            _assert_pass_criteria(
                config,
                model_env=model_env,
                work_dir=work_dir,
                eval_log_path=eval_log_path,
            )
        finally:
            if (
                os.environ.get("VLLM_HCU_KEEP_SERVER_ON_FAILURE") == "1"
                and sys.exc_info()[0] is not None
            ):
                print(f"keeping vLLM server alive for debugging: pid={proc.pid}")
            else:
                _terminate_process_group(proc, shutdown_timeout)
