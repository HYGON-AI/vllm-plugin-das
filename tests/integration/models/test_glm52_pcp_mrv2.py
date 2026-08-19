# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eight-HCU GLM-5.2 model-runner-v2 PCP acceptance coverage."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import require_model_runtime
from tests.integration.server.evalscope_server import (
    ROOT,
    _open_log,
    _server_environment,
    _terminate_process_group,
    _wait_for_server,
    load_config,
    server_command,
)
from tests.integration.server.openai_server import OpenAIServer


DEFAULT_MODEL = "/models/GLM-5___1-Channel-FP8-w8a8"
MODEL_ENV = "VLLM_HCU_GLM52_MODEL"
DEFAULT_CONFIG = ROOT / "tests/models/glm52_pcp_humaneval_evalscope.yaml"


@dataclass(frozen=True)
class CandidateServer:
    api: OpenAIServer
    process: subprocess.Popen[bytes]
    artifact_dir: Path


def _chat_payload(request_id: str) -> dict[str, Any]:
    return {
        "model": "GLM-5___1-Channel-FP8-w8a8",
        "request_id": request_id,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly one word naming Earth's satellite.",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _completion_payload(
    prompt_token_ids: list[int], request_id: str
) -> dict[str, Any]:
    return {
        "model": "GLM-5___1-Channel-FP8-w8a8",
        "request_id": request_id,
        "prompt": prompt_token_ids,
        "temperature": 0,
        "max_tokens": 8,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_glm52_smoke_payload_contract() -> None:
    payload = _chat_payload("glm52-pcp-smoke-01")

    assert payload["request_id"] == "glm52-pcp-smoke-01"
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["return_token_ids"] is True


@contextmanager
def _serve_candidate(
    resources: HcuTestResources,
) -> Iterator[CandidateServer]:
    require_model_runtime(
        resources,
        env_name=MODEL_ENV,
        relative_path=DEFAULT_MODEL,
        label="GLM-5.2 TP4+PCP2+EP",
        hcu_count=8,
    )
    config = load_config(DEFAULT_CONFIG, "VLLM_HCU_GLM52_HUMANEVAL_CONFIG")
    command, host, port = server_command(config, model_env=MODEL_ENV)
    environment = _server_environment(config)
    artifact_dir = Path(
        os.environ.get(
            "VLLM_HCU_TEST_ARTIFACT_DIR",
            "/tmp/vllm-hcu-integration/glm52-pcp",
        )
    )
    server_log_path = artifact_dir / "vllm_server.log"
    with _open_log(server_log_path) as server_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(
                process,
                f"http://{host}:{port}/health",
                int(config["server"]["startup_timeout_s"]),
            )
            yield CandidateServer(
                api=OpenAIServer(
                    base_url=f"http://{host}:{port}",
                    model_name="GLM-5___1-Channel-FP8-w8a8",
                    log_path=server_log_path,
                ),
                process=process,
                artifact_dir=artifact_dir,
            )
        finally:
            _terminate_process_group(
                process,
                int(config["server"]["shutdown_timeout_s"]),
            )


@pytest.fixture(scope="module")
def glm52_candidate_server(
    hcu_test_resources: HcuTestResources,
) -> Iterator[CandidateServer]:
    with _serve_candidate(hcu_test_resources) as candidate:
        yield candidate


def _peak_device_memory_sampler(stop: threading.Event, samples: list[int]) -> None:
    import torch

    while not stop.wait(0.05):
        usage = 0
        for device_index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            usage += total_bytes - free_bytes
        samples.append(usage)


def _stream_completion(
    candidate: CandidateServer,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = Request(
        candidate.api.base_url + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_peak_device_memory_sampler,
        args=(stop, samples),
        daemon=True,
    )
    started = time.perf_counter()
    sampler.start()
    token_ids: list[int] = []
    prompt_token_ids: list[int] | None = None
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    response_id: str | None = None
    try:
        with urlopen(request, timeout=7200) as response:
            assert response.status == 200
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                response_id = response_id or chunk.get("id")
                if prompt_token_ids is None and chunk.get("prompt_token_ids"):
                    prompt_token_ids = list(chunk["prompt_token_ids"])
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                for choice in chunk.get("choices", []):
                    if prompt_token_ids is None and choice.get("prompt_token_ids"):
                        prompt_token_ids = list(choice["prompt_token_ids"])
                    delta = list(choice.get("token_ids") or [])
                    if delta and first_token_at is None:
                        first_token_at = time.perf_counter()
                    token_ids.extend(delta)
    finally:
        finished = time.perf_counter()
        stop.set()
        sampler.join(timeout=5)

    assert response_id is not None
    assert response_id.endswith(payload["request_id"])
    assert token_ids, "the request must decode at least one token"
    assert first_token_at is not None
    assert candidate.process.poll() is None, "a PCP worker rank exited during decode"
    latency = finished - started
    prompt_count = int(usage.get("prompt_tokens", len(prompt_token_ids or [])))
    completion_count = int(usage.get("completion_tokens", len(token_ids)))
    return {
        "request_id": payload["request_id"],
        "prompt_token_count": prompt_count,
        "completion_token_count": completion_count,
        "token_ids": token_ids,
        "latency_seconds": latency,
        "ttft_seconds": first_token_at - started,
        "throughput_tokens_per_second": (prompt_count + completion_count) / latency,
        "peak_memory_bytes": max(samples, default=0),
    }


def _write_metrics(candidate: CandidateServer, case_id: str, record: dict) -> None:
    candidate.artifact_dir.mkdir(parents=True, exist_ok=True)
    (candidate.artifact_dir / f"{case_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("request_id", ["glm52-pcp-smoke-01"])
@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
def test_glm52_pcp_deterministic_smoke(
    glm52_candidate_server: CandidateServer,
    request_id: str,
) -> None:
    payload = _chat_payload(request_id)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    first = _stream_completion(
        glm52_candidate_server,
        "/v1/chat/completions",
        payload,
    )
    second_payload = {**payload, "request_id": "glm52-pcp-smoke-02"}
    second = _stream_completion(
        glm52_candidate_server,
        "/v1/chat/completions",
        second_payload,
    )

    rank_converged = (
        glm52_candidate_server.process.poll() is None
        and first["token_ids"] == second["token_ids"]
    )
    assert rank_converged
    _write_metrics(
        glm52_candidate_server,
        request_id,
        {
            "first": first,
            "second": second,
            "rank_divergence": not rank_converged,
        },
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "request_id"),
    [
        pytest.param(32768, "glm52-pcp-context-32k", id="32k-prefill"),
        pytest.param(65536, "glm52-pcp-context-64k", id="64k-prefill"),
    ],
)
@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
def test_glm52_pcp_long_context_decodes_after_prefill(
    glm52_candidate_server: CandidateServer,
    prompt_tokens: int,
    request_id: str,
) -> None:
    tokenized = glm52_candidate_server.api.post(
        "/tokenize",
        {
            "model": glm52_candidate_server.api.model_name,
            "prompt": "GLM PCP deterministic long-context token.",
        },
    )
    assert tokenized.status == 200
    seed_token_ids = tokenized.body["tokens"]
    assert seed_token_ids
    repeated = (seed_token_ids * (prompt_tokens // len(seed_token_ids) + 1))[
        :prompt_tokens
    ]

    record = _stream_completion(
        glm52_candidate_server,
        "/v1/completions",
        _completion_payload(repeated, request_id),
    )

    assert record["prompt_token_count"] == prompt_tokens
    assert record["completion_token_count"] >= 1
    assert record["peak_memory_bytes"] > 0
    _write_metrics(glm52_candidate_server, request_id, record)
