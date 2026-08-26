# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reusable real-vLLM server process for protocol integration tests."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = Path("/tmp/vllm-hcu-integration/logs")


def _require_structured_output_dependencies() -> None:
    required = {
        "lmformatenforcer": "lm-format-enforcer==0.11.3",
        "outlines_core": "outlines_core==0.2.14",
    }
    missing = [
        requirement
        for module, requirement in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "structured-output protocol tests require: "
            + " ".join(missing)
            + "; install with `python -m pip install -r requirements-test.txt`"
        )


def _available_port() -> int:
    configured = os.environ.get("VLLM_HCU_PROTOCOL_SERVER_PORT")
    if configured is not None:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _log_tail(log_path: Path, max_bytes: int = 16 * 1024) -> str:
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read().decode(errors="replace")
    except OSError as exc:
        return f"<unable to read server log: {exc}>"


def _wait_for_health(
    proc: subprocess.Popen,
    url: str,
    timeout_s: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM protocol server exited before ready, rc={returncode}; "
                f"log={log_path}\nserver log tail:\n{_log_tail(log_path)}"
            )
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(
        f"vLLM protocol server was not ready at {url}: {last_error}; "
        f"log={log_path}\nserver log tail:\n{_log_tail(log_path)}"
    )


def _terminate_process_group(proc: subprocess.Popen, timeout_s: int = 60) -> None:
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


@dataclass(frozen=True)
class ProtocolResponse:
    status: int
    body: dict[str, Any]


@dataclass(frozen=True)
class ProtocolTextResponse:
    status: int
    text: str


@dataclass(frozen=True)
class ProtocolStreamEvent:
    event: str | None
    data: dict[str, Any] | str


@dataclass(frozen=True)
class ProtocolStreamResponse:
    status: int
    events: list[ProtocolStreamEvent]


def _parse_sse_events(raw: bytes) -> list[ProtocolStreamEvent]:
    events: list[ProtocolStreamEvent] = []
    event_type: str | None = None
    data_lines: list[str] = []

    def emit() -> None:
        nonlocal event_type
        if not data_lines:
            event_type = None
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        if payload == "[DONE]":
            data: dict[str, Any] | str = payload
        else:
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    "stream response contained invalid SSE JSON"
                ) from exc
            if not isinstance(decoded, dict):
                raise AssertionError("stream response event was not a JSON object")
            data = decoded
        events.append(ProtocolStreamEvent(event=event_type, data=data))
        event_type = None

    for line in raw.decode(errors="replace").splitlines():
        if not line:
            emit()
        elif line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    emit()
    if not events:
        raise AssertionError("stream response did not contain SSE events")
    return events


@dataclass(frozen=True)
class OpenAIServer:
    base_url: str
    model_name: str
    log_path: Path

    def log_tail(self) -> str:
        return _log_tail(self.log_path)

    def get_text(
        self,
        path: str,
        *,
        timeout_s: int = 30,
    ) -> ProtocolTextResponse:
        request = Request(self.base_url + path, method="GET")
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return ProtocolTextResponse(
                    status=response.status,
                    text=response.read().decode(errors="replace"),
                )
        except HTTPError as exc:
            return ProtocolTextResponse(
                status=exc.code,
                text=exc.read().decode(errors="replace"),
            )

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout_s: int = 180,
    ) -> ProtocolResponse:
        request_headers = {
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers=request_headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                status = response.status
                raw = response.read()
        except HTTPError as exc:
            status = exc.code
            raw = exc.read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{path} returned non-JSON status={status}: {raw[:1000]!r}; "
                f"server_log={self.log_path}"
            ) from exc
        if not isinstance(body, dict):
            raise AssertionError(
                f"{path} returned non-object JSON status={status}: {body!r}; "
                f"server_log={self.log_path}"
            )
        return ProtocolResponse(status=status, body=body)

    def post_sse(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout_s: int = 180,
    ) -> ProtocolStreamResponse:
        request_headers = {
            "Accept": "text/event-stream",
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers=request_headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                status = response.status
                raw = response.read()
        except HTTPError as exc:
            status = exc.code
            raw = exc.read()
        if status != 200:
            raise AssertionError(
                f"{path} stream request failed with status={status}; "
                f"server_log={self.log_path}"
            )
        return ProtocolStreamResponse(status=status, events=_parse_sse_events(raw))


@contextmanager
def serve_openai_protocol_model(
    model_path: Path,
    *,
    startup_timeout_s: int = 1800,
    enable_qwen3_parsers: bool = False,
    extra_args: list[str] | None = None,
) -> Iterator[OpenAIServer]:
    if enable_qwen3_parsers:
        _require_structured_output_dependencies()
    port = _available_port()
    model_name = model_path.name
    gpu_memory_utilization = os.environ.get(
        "VLLM_HCU_PROTOCOL_GPU_MEMORY_UTILIZATION",
        "0.2",
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_path),
        "--served-model-name",
        model_name,
        "--trust-remote-code",
        "--enforce-eager",
        "--log-error-stack",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--port",
        str(port),
    ]
    if enable_qwen3_parsers:
        command[5:5] = [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser",
            "qwen3",
        ]
    if extra_args:
        command.extend(extra_args)
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if os.environ.get("VLLM_HCU_RELEASE_WHEEL") == "1":
        server_cwd = Path(
            os.environ.get("HCU_CI_JOB_ROOT", "/tmp/vllm-hcu-release-wheel")
        ) / "server-subprocess"
        server_cwd.mkdir(parents=True, exist_ok=True)
    else:
        server_cwd = ROOT
    log_dir = Path(
        os.environ.get("VLLM_HCU_INTEGRATION_LOG_DIR", DEFAULT_LOG_DIR)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{model_name}_openai-protocol.log"
    with log_path.open("ab") as log:
        log.write(("server command: " + " ".join(command) + "\n").encode())
        log.write(b"server environment: VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=server_cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            _wait_for_health(
                proc,
                base_url + "/health",
                startup_timeout_s,
                log_path,
            )
            yield OpenAIServer(
                base_url=base_url,
                model_name=model_name,
                log_path=log_path,
            )
        finally:
            _terminate_process_group(proc)
