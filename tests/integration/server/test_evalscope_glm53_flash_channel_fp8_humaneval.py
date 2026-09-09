# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3-Flash Channel-FP8 TP4 OpenAI/HumanEval acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    evalscope_command,
    load_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT / "tests/models/glm53_flash_channel_fp8_humaneval_evalscope.yaml"
)
CONFIG_ENV = "VLLM_HCU_GLM53_FLASH_CHANNEL_FP8_HUMANEVAL_CONFIG"
MODEL_ENV = "VLLM_HCU_GLM53_FLASH_CHANNEL_FP8_MODEL"


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_glm53_flash_channel_fp8_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)

    server, host, port = server_command(config, model_env=MODEL_ENV)
    evaluation = evalscope_command(
        config,
        model_env=MODEL_ENV,
        host=host,
        port=port,
        work_dir=tmp_path,
    )

    assert server[:3] == [
        "vllm",
        "serve",
        "/models/GLM-5.3-Flash-Channel-FP8-w8a8",
    ]
    assert _option_value(server, "--tensor-parallel-size") == "4"
    assert _option_value(server, "--attention-backend") == "FLASHMLA_SPARSE"
    assert _option_value(server, "--moe-backend") == "aiter"
    assert json.loads(_option_value(server, "--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    assert json.loads(_option_value(server, "--compilation-config")) == {
        "cudagraph_mode": "PIECEWISE"
    }
    assert _option_value(server, "--max-model-len") == "4096"
    assert _option_value(server, "--max-num-batched-tokens") == "1024"
    assert _option_value(server, "--max-num-seqs") == "8"
    assert "--enforce-eager" not in server
    assert "--quantization" not in server
    assert json.loads(
        _option_value(server, "--default-chat-template-kwargs")
    ) == {"reasoning_effort": "low"}
    assert _option_value(evaluation, "--datasets") == "humaneval"
    assert _option_value(evaluation, "--limit") == "8"
    assert _option_value(evaluation, "--eval-batch-size") == "8"
    assert _option_value(evaluation, "--model") == _option_value(
        server,
        "--served-model-name",
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(4)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_glm53_flash_channel_fp8_humaneval_evalscope_server() -> None:
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="GLM-5.3-Flash Channel-FP8 TP4+AITER+MTP3",
        required_hcu_count=4,
    )
