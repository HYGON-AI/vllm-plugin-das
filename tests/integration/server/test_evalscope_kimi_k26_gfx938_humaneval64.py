# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K2.6 gfx938 TP8 OpenAI/HumanEval64 diagnostic gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    evalscope_command,
    load_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/models/kimi_k26_gfx938_humaneval64.yaml"
CONFIG_ENV = "VLLM_HCU_KIMI_K26_HUMANEVAL64_CONFIG"
MODEL_ENV = "VLLM_HCU_KIMI_K26_MODEL"


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_kimi_k26_gfx938_humaneval64_command_contract(
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

    assert server[:3] == ["vllm", "serve", "/models/Kimi-K2.6"]
    assert config["server"]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert "--language-model-only" in server
    assert _option_value(server, "--tensor-parallel-size") == "8"
    assert _option_value(server, "--attention-backend") == "FLASHMLA"
    assert _option_value(server, "--moe-backend") == "triton"
    assert _option_value(server, "--reasoning-parser") == "kimi_k2"
    assert "--kv-cache-dtype" not in server
    assert "--enable-prefix-caching" in server
    assert "--enforce-eager" not in server
    assert "--compilation-config" not in server
    assert "--speculative-config" not in server
    assert "--default-chat-template-kwargs" not in server
    assert _option_value(evaluation, "--datasets") == "humaneval"
    assert _option_value(evaluation, "--limit") == "64"
    assert _option_value(evaluation, "--eval-batch-size") == "8"
    assert config["evalscope"]["generation_config"]["temperature"] == 0.6
    assert config["evalscope"]["generation_config"]["top_p"] == 0.95
    assert config["evalscope"]["generation_config"]["do_sample"] is True
    assert config["evalscope"]["generation_config"]["max_tokens"] == 4096
    assert config["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"thinking": False}
    }
    assert config["evalscope"]["pass_criteria"] == {
        "dataset": "humaneval",
        "num_predictions": 64,
        "num_reviews": 64,
        "normalize_code_fences": True,
        "record_normalized_score": True,
        "enforce_score": False,
    }


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_kimi_k26_gfx938_humaneval64() -> None:
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="Kimi-K2.6 gfx938 TP8",
        required_hcu_count=8,
    )
