# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared P/D-disaggregated vLLM + EvalScope integration-test runner."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from tests.integration.server.evalscope_server import (
    _assert_pass_criteria,
    _direct_urlopen,
    _open_log,
    _owned_process_environment,
    _require_runtime,
    _reset_evalscope_artifacts,
    _server_environment,
    _terminate_process_group,
    _wait_for_server,
    evalscope_command,
)


ROOT = Path(__file__).resolve().parents[3]


_REQUIRED_PREFILL_EVIDENCE = (
    "Mooncake TTFT_EVENT event=p_send_kv_done",
    "DeepEP auto selected contiguous high-throughput experts for this forward.",
    "Using DeepEPDeepGemmContiguousExperts with DeepGEMM HT path.",
)
_REQUIRED_DECODE_EVIDENCE = (
    "Mooncake TTFT_EVENT event=d_kv_ready",
    "DeepEP auto selected masked low-latency experts for this forward.",
    "Using DeepEPDeepGemmMaskedExperts with DeepGEMM LL path.",
)
_MOONCAKE_FAILURES = (
    re.compile(r"Sending to .* failed"),
    re.compile(r"MooncakeXferMetadata transfer failed"),
)


@dataclass(frozen=True)
class PDCommands:
    """Commands, environments, and endpoints for one P/D test topology."""

    prefill: list[str]
    decode: list[str]
    proxy: list[str]
    prefill_env: dict[str, str]
    decode_env: dict[str, str]
    proxy_env: dict[str, str]
    host: str
    proxy_port: int
    prefill_port: int
    decode_port: int
    startup_timeout_s: int
    shutdown_timeout_s: int


def _prometheus_sum(metrics: str, metric_name: str) -> float:
    sample = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"(?:\s+\d+)?\s*$",
        re.MULTILINE,
    )
    return sum(float(match.group(1)) for match in sample.finditer(metrics))


def assert_pd_runtime_evidence(
    prefill_log: Path,
    decode_log: Path,
    decode_metrics: str,
) -> None:
    """Require evidence that Mooncake, DeepEP/DeepGEMM, and DSpark ran."""

    prefill_text = prefill_log.read_text(encoding="utf-8", errors="replace")
    decode_text = decode_log.read_text(encoding="utf-8", errors="replace")
    combined = f"{prefill_text}\n{decode_text}"

    for failure in _MOONCAKE_FAILURES:
        assert failure.search(combined) is None, (
            f"Mooncake transfer failure found in P/D logs: {failure.pattern}"
        )
    for marker in _REQUIRED_PREFILL_EVIDENCE:
        assert marker in prefill_text, f"missing prefill runtime evidence: {marker}"
    for marker in _REQUIRED_DECODE_EVIDENCE:
        assert marker in decode_text, f"missing decode runtime evidence: {marker}"

    draft_tokens = _prometheus_sum(
        decode_metrics,
        "vllm:spec_decode_num_draft_tokens_total",
    )
    accepted_tokens = _prometheus_sum(
        decode_metrics,
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    assert draft_tokens > 0, f"DSpark draft tokens must be positive, got {draft_tokens}"
    assert accepted_tokens > 0, (
        f"DSpark accepted tokens must be positive, got {accepted_tokens}"
    )


def _wait_for_routed_smoke(
    proc: subprocess.Popen,
    url: str,
    model: str,
    timeout_s: int,
) -> None:
    """Wait until the proxy can route a real completion through P and D."""

    payload = json.dumps(
        {
            "model": model,
            "prompt": "Return the word ready.",
            "temperature": 0,
            "max_tokens": 16,
        }
    ).encode()
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"Mooncake proxy exited before routed smoke, rc={returncode}"
            )
        request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _direct_urlopen(request, timeout=30) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(
                    f"routed smoke returned HTTP {response.status}"
                )
        except HTTPError as exc:
            if exc.code != 503:
                raise RuntimeError(
                    f"routed smoke returned HTTP {exc.code}"
                ) from exc
            last_error = exc
        except URLError as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(
        f"Mooncake proxy did not complete routed smoke at {url}: {last_error}"
    )


def _fetch_text(url: str, timeout: int) -> str:
    with _direct_urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _reset_pd_logs(log_dir: Path) -> None:
    """Remove only the owned P/D acceptance evidence files."""

    log_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "prefill.log",
        "decode.log",
        "proxy.log",
        "evalscope.log",
        "decode_metrics.prom",
    ):
        path = log_dir / name
        if path.is_dir() and not path.is_symlink():
            raise ValueError(f"P/D log path must not be a directory: {path}")
        path.unlink(missing_ok=True)


def _role_command(
    *,
    model: str,
    served_model_name: str,
    common_args: list[str],
    role: dict[str, Any],
) -> list[str]:
    return [
        "vllm",
        "serve",
        model,
        *common_args,
        *(str(item) for item in role.get("args", [])),
        "--served-model-name",
        served_model_name,
        "--port",
        str(role["port"]),
        "--data-parallel-rpc-port",
        str(role["data_parallel_rpc_port"]),
    ]


def _role_environment(
    base: dict[str, str],
    *,
    host: str,
    role: dict[str, Any],
) -> dict[str, str]:
    env = base.copy()
    env.update(
        {
            "HIP_VISIBLE_DEVICES": str(role["visible_devices"]),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(role["bootstrap_port"]),
            "VLLM_DP_MASTER_IP": host,
            "VLLM_DP_MASTER_PORT": str(role["data_parallel_master_port"]),
        }
    )
    return env


def pd_commands(config: dict[str, Any], *, model_env: str) -> PDCommands:
    """Build the official-style P, D, and Mooncake proxy commands."""

    pd = config["pd"]
    host = str(pd.get("host", "127.0.0.1"))
    model = os.environ.get(model_env, str(config["model"]))
    served_model_name = str(config["served_model_name"])
    common_args = [str(item) for item in pd.get("common_args", [])]
    prefill = pd["prefill"]
    decode = pd["decode"]

    base_env = _server_environment()
    environment = pd.get("environment", {})
    if not isinstance(environment, dict):
        raise TypeError("pd.environment must be a mapping")
    base_env.update({str(name): str(value) for name, value in environment.items()})

    source_root_value = os.environ.get("VLLM_V0251_SOURCE_ROOT")
    if source_root_value is None:
        raise FileNotFoundError(
            "VLLM_V0251_SOURCE_ROOT must identify the vLLM v0.25.1 source tree"
        )
    proxy_script = (
        Path(source_root_value)
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    if not proxy_script.is_file():
        raise FileNotFoundError(f"Mooncake connector proxy is absent: {proxy_script}")

    proxy_port = int(pd["proxy_port"])
    prefill_port = int(prefill["port"])
    decode_port = int(decode["port"])
    proxy = [
        sys.executable,
        str(proxy_script),
        "--prefill",
        f"http://{host}:{prefill_port}",
        str(prefill["bootstrap_port"]),
        "--decode",
        f"http://{host}:{decode_port}",
        "--host",
        host,
        "--port",
        str(proxy_port),
    ]

    return PDCommands(
        prefill=_role_command(
            model=model,
            served_model_name=served_model_name,
            common_args=common_args,
            role=prefill,
        ),
        decode=_role_command(
            model=model,
            served_model_name=served_model_name,
            common_args=common_args,
            role=decode,
        ),
        proxy=proxy,
        prefill_env=_role_environment(base_env, host=host, role=prefill),
        decode_env=_role_environment(base_env, host=host, role=decode),
        proxy_env=base_env.copy(),
        host=host,
        proxy_port=proxy_port,
        prefill_port=prefill_port,
        decode_port=decode_port,
        startup_timeout_s=int(pd.get("startup_timeout_s", 3600)),
        shutdown_timeout_s=int(pd.get("shutdown_timeout_s", 60)),
    )


def run_evalscope_pd_server_test(
    config: dict[str, Any],
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> None:
    """Run owned P, D, proxy, and EvalScope processes with evidence gates."""

    _require_runtime(
        config,
        model_env=model_env,
        model_label=model_label,
        required_hcu_count=required_hcu_count,
    )
    commands = pd_commands(config, model_env=model_env)
    startup_timeout = int(
        os.environ.get(
            "VLLM_HCU_SERVER_STARTUP_TIMEOUT",
            commands.startup_timeout_s,
        )
    )
    work_dir = Path(
        os.environ.get("VLLM_HCU_EVAL_WORK_DIR", config["evalscope"]["work_dir"])
    )
    _reset_evalscope_artifacts(work_dir)
    log_dir = work_dir / "logs"
    _reset_pd_logs(log_dir)
    prefill_log_path = log_dir / "prefill.log"
    decode_log_path = log_dir / "decode.log"
    proxy_log_path = log_dir / "proxy.log"
    eval_log_path = log_dir / "evalscope.log"
    decode_metrics_path = log_dir / "decode_metrics.prom"

    prefill_proc: subprocess.Popen | None = None
    decode_proc: subprocess.Popen | None = None
    proxy_proc: subprocess.Popen | None = None
    prefill_owner: str | None = None
    decode_owner: str | None = None
    proxy_owner: str | None = None
    prefill_log = _open_log(prefill_log_path)
    decode_log = None
    proxy_log = None
    try:
        prefill_log.write(
            ("prefill command: " + " ".join(commands.prefill) + "\n").encode()
        )
        prefill_log.flush()
        prefill_env, prefill_owner = _owned_process_environment(
            commands.prefill_env
        )
        prefill_proc = subprocess.Popen(
            commands.prefill,
            cwd=ROOT,
            env=prefill_env,
            stdout=prefill_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_server(
            prefill_proc,
            f"http://{commands.host}:{commands.prefill_port}/health",
            startup_timeout,
        )

        decode_log = _open_log(decode_log_path)
        decode_log.write(
            ("decode command: " + " ".join(commands.decode) + "\n").encode()
        )
        decode_log.flush()
        decode_env, decode_owner = _owned_process_environment(commands.decode_env)
        decode_proc = subprocess.Popen(
            commands.decode,
            cwd=ROOT,
            env=decode_env,
            stdout=decode_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_server(
            decode_proc,
            f"http://{commands.host}:{commands.decode_port}/health",
            startup_timeout,
        )

        proxy_log = _open_log(proxy_log_path)
        proxy_log.write(
            ("proxy command: " + " ".join(commands.proxy) + "\n").encode()
        )
        proxy_log.flush()
        proxy_env, proxy_owner = _owned_process_environment(commands.proxy_env)
        proxy_proc = subprocess.Popen(
            commands.proxy,
            cwd=ROOT,
            env=proxy_env,
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_routed_smoke(
            proxy_proc,
            f"http://{commands.host}:{commands.proxy_port}/v1/completions",
            str(config["served_model_name"]),
            startup_timeout,
        )

        eval_config = copy.deepcopy(config)
        eval_config["server"] = {
            "served_model_name": str(config["served_model_name"])
        }
        eval_command = evalscope_command(
            eval_config,
            model_env=model_env,
            host=commands.host,
            port=commands.proxy_port,
            work_dir=work_dir,
        )
        with _open_log(eval_log_path) as eval_log:
            eval_log.write(
                ("eval command: " + " ".join(eval_command) + "\n").encode()
            )
            eval_log.flush()
            result = subprocess.run(
                eval_command,
                cwd=ROOT,
                env=commands.proxy_env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"EvalScope failed with rc={result.returncode}; log={eval_log_path}"
            )

        decode_metrics = _fetch_text(
            f"http://{commands.host}:{commands.decode_port}/metrics",
            30,
        )
        decode_metrics_path.write_text(decode_metrics, encoding="utf-8")
        _assert_pass_criteria(
            eval_config,
            model_env=model_env,
            work_dir=work_dir,
            eval_log_path=eval_log_path,
        )
        prefill_log.flush()
        if decode_log is not None:
            decode_log.flush()
        assert_pd_runtime_evidence(
            prefill_log_path,
            decode_log_path,
            decode_metrics,
        )
    finally:
        for proc, owner_token in (
            (proxy_proc, proxy_owner),
            (decode_proc, decode_owner),
            (prefill_proc, prefill_owner),
        ):
            if proc is not None:
                _terminate_process_group(
                    proc,
                    commands.shutdown_timeout_s,
                    owner_token=owner_token,
                )
        if proxy_log is not None:
            proxy_log.close()
        if decode_log is not None:
            decode_log.close()
        prefill_log.close()
