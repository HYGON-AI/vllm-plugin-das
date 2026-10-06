# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K2.5 OpenAI server and HumanEval acceptance test."""

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
DEFAULT_CONFIG = ROOT / "tests/models/kimi_k25_humaneval_evalscope.yaml"
CONFIG_ENV = "VLLM_HCU_KIMI_K25_HUMANEVAL_CONFIG"
MODEL_ENV = "VLLM_HCU_KIMI_K25_MODEL"


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_kimi_k25_humaneval_uses_official_whitespace_filter(
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

    assert server[:3] == ["vllm", "serve", "/model/kimi-K2.5"]
    assert _option_value(server, "--attention-backend") == "FLASHMLA"
    assert _option_value(server, "--moe-backend") == "aiter"
    assert _option_value(server, "--kv-cache-dtype") == "fp8_e5m2"
    assert _option_value(server, "--tool-call-parser") == "kimi_k2"
    assert "--enable-auto-tool-choice" in server
    assert _option_value(server, "--reasoning-parser") == "kimi_k2"
    assert "--enforce-eager" not in server
    assert json.loads(_option_value(evaluation, "--dataset-args")) == {
        "humaneval": {"filters": {"remove_whitespace": {}}}
    }


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_kimi_k25_humaneval_evalscope_server() -> None:
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="Kimi-K2.5 TP8+EP8 W4A16 E5M2",
        required_hcu_count=8,
    )
