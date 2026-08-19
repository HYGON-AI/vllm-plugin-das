# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.2 TP4+PCP2+EP OpenAI server and HumanEval acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    _server_environment,
    evalscope_command,
    load_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/models/glm52_pcp_humaneval_evalscope.yaml"
MODEL_ENV = "VLLM_HCU_GLM52_MODEL"


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_glm52_pcp_server_command_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV, raising=False)
    monkeypatch.delenv("VLLM_HCU_GLM52_HUMANEVAL_CONFIG", raising=False)
    config = load_config(DEFAULT_CONFIG, "VLLM_HCU_GLM52_HUMANEVAL_CONFIG")

    command, host, port = server_command(config, model_env=MODEL_ENV)
    environment = _server_environment(config)

    assert command[:3] == [
        "vllm",
        "serve",
        "/models/GLM-5___1-Channel-FP8-w8a8",
    ]
    assert host == "127.0.0.1"
    assert port == 10132
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert _option_value(command, "--tensor-parallel-size") == "4"
    assert _option_value(command, "--prefill-context-parallel-size") == "2"
    assert "--enable-expert-parallel" in command
    assert "--enforce-eager" in command
    assert _option_value(command, "--moe-backend") == "triton"
    assert _option_value(command, "--max-model-len") == "69632"
    assert (
        _option_value(command, "--served-model-name")
        == "GLM-5___1-Channel-FP8-w8a8"
    )
    assert json.loads(
        _option_value(command, "--default-chat-template-kwargs")
    ) == {"enable_thinking": False}

    forbidden_options = {
        "--speculative-config",
        "--speculative-model",
        "--num-speculative-tokens",
        "--pipeline-parallel-size",
        "--decode-context-parallel-size",
        "--compilation-config",
        "--quantization",
    }
    assert forbidden_options.isdisjoint(command)

    monkeypatch.setenv(MODEL_ENV, "/models/overrides/glm52")
    override_command, _, _ = server_command(config, model_env=MODEL_ENV)
    assert override_command[2] == "/models/overrides/glm52"


def test_glm52_humaneval_config_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_GLM52_HUMANEVAL_CONFIG", raising=False)
    config = load_config(DEFAULT_CONFIG, "VLLM_HCU_GLM52_HUMANEVAL_CONFIG")
    evalscope = config["evalscope"]

    assert evalscope["datasets"] == ["humaneval"]
    assert evalscope["limit"] == 32
    assert evalscope["eval_type"] == "openai_api"
    assert evalscope["eval_batch_size"] == 1
    assert evalscope["generation_config"] == {
        "temperature": 0,
        "do_sample": False,
        "max_tokens": 2048,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_glm52_humaneval_request_payload_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_GLM52_HUMANEVAL_CONFIG", raising=False)
    config = load_config(DEFAULT_CONFIG, "VLLM_HCU_GLM52_HUMANEVAL_CONFIG")

    command = evalscope_command(
        config,
        model_env=MODEL_ENV,
        host="127.0.0.1",
        port=10132,
        work_dir=tmp_path,
    )
    generation = json.loads(_option_value(command, "--generation-config"))

    assert generation["temperature"] == 0
    assert generation["do_sample"] is False
    assert generation["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert _option_value(command, "--limit") == "32"
    assert _option_value(command, "--eval-batch-size") == "1"
    assert _option_value(command, "--datasets") == "humaneval"


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_glm52_pcp_humaneval_evalscope_server() -> None:
    config = load_config(DEFAULT_CONFIG, "VLLM_HCU_GLM52_HUMANEVAL_CONFIG")
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="GLM-5.2 TP4+PCP2+EP",
        required_hcu_count=8,
    )
