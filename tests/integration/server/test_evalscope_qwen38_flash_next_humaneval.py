# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Qwen3.8 Flash Next OpenAI server and HumanEval acceptance test."""

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
    ROOT / "tests/models/qwen38_flash_next_humaneval_evalscope.yaml"
)
CONFIG_ENV = "VLLM_HCU_QWEN38_FLASH_NEXT_HUMANEVAL_CONFIG"
MODEL_ENV = "VLLM_HCU_QWEN38_FLASH_NEXT_MODEL"


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_qwen38_flash_next_humaneval_command_contract(
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
        "/models/Qwen3.8-Flash-Next-FP8-Channelwise",
    ]
    assert _option_value(server, "--tensor-parallel-size") == "4"
    assert "--enforce-eager" not in server
    assert _option_value(server, "--attention-backend") == "FLASH_ATTN"
    assert _option_value(server, "--moe-backend") == "aiter"
    assert _option_value(server, "--spec-method") == "mtp"
    assert _option_value(server, "--spec-tokens") == "3"
    assert "--enable-prefix-caching" in server
    assert _option_value(server, "--max-num-seqs") == "4"
    assert _option_value(server, "--max-model-len") == "40960"
    assert json.loads(
        _option_value(server, "--default-chat-template-kwargs")
    ) == {"enable_thinking": False}
    assert _option_value(evaluation, "--datasets") == "humaneval"
    assert _option_value(evaluation, "--limit") == "8"
    assert _option_value(evaluation, "--eval-batch-size") == "1"
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
def test_qwen38_flash_next_humaneval_evalscope_server() -> None:
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="Qwen3.8 Flash Next TP4+AITER+MTP3",
        required_hcu_count=4,
    )
