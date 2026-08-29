# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4-Flash-0731 Channel-FP8/INT8 HumanEval-32 acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    evalscope_command,
    load_profiled_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/models/deepseek_v4_flash_0731_dspark_humaneval.yaml"
CONFIG_ENV = "VLLM_HCU_DEEPSEEK_V4_DSPARK_HUMANEVAL_CONFIG"
MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL"
INT8_CONFIG = (
    ROOT / "tests/models/deepseek_v4_flash_0731_int8_dspark_humaneval.yaml"
)
INT8_CONFIG_ENV = "VLLM_HCU_DEEPSEEK_V4_INT8_DSPARK_HUMANEVAL_CONFIG"
INT8_MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL"
DSPARK_CONFIG = {
    "method": "dspark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
}


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _load(profile: str) -> dict:
    return load_profiled_config(
        DEFAULT_CONFIG,
        CONFIG_ENV,
        profile=profile,
    )


def _load_int8(profile: str) -> dict:
    return load_profiled_config(
        INT8_CONFIG,
        INT8_CONFIG_ENV,
        profile=profile,
    )


@pytest.mark.parametrize("profile", ["tp8", "dp8_ep8"])
def test_deepseek_v4_int8_dspark_humaneval_server_contract(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.delenv(INT8_CONFIG_ENV, raising=False)
    monkeypatch.delenv(INT8_MODEL_ENV, raising=False)
    config = _load_int8(profile)
    command, _, _ = server_command(config, model_env=INT8_MODEL_ENV)

    assert command[:3] == [
        "vllm",
        "serve",
        "/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8",
    ]
    assert _option_value(command, "--kv-cache-dtype") == "fp8"
    assert json.loads(_option_value(command, "--speculative-config")) == (
        DSPARK_CONFIG
    )
    assert config["evalscope"]["limit"] == 32
    assert config["evalscope"]["pass_criteria"]["mean_acc"] == 1.0
    assert config["evalscope"]["pass_criteria"]["mean_acc_pass@1"] == 1.0
    assert "--enforce-eager" not in command
    assert "--moe-backend" not in command
    assert not any(
        fragment in item
        for item in command
        for fragment in (
            "prefill-context-parallel",
            "kv-transfer",
            "deepep_high_throughput",
            "deepep_low_latency",
        )
    )
    if profile == "tp8":
        assert _option_value(command, "--tensor-parallel-size") == "8"
        assert "--data-parallel-size" not in command
        assert "--enable-expert-parallel" not in command
        assert "--all2all-backend" not in command
    else:
        assert _option_value(command, "--tensor-parallel-size") == "1"
        assert _option_value(command, "--data-parallel-size") == "8"
        assert "--enable-expert-parallel" in command
        assert _option_value(command, "--all2all-backend") == "deepep_auto"


@pytest.mark.parametrize("profile", ["tp8", "dp8_ep8"])
def test_deepseek_v4_dspark_humaneval_common_server_contract(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    command, _, _ = server_command(_load(profile), model_env=MODEL_ENV)

    assert command[:3] == [
        "vllm",
        "serve",
        "/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
    ]
    assert _option_value(command, "--tokenizer-mode") == "deepseek_v4"
    assert _option_value(command, "--kv-cache-dtype") == "fp8"
    assert json.loads(_option_value(command, "--speculative-config")) == (
        DSPARK_CONFIG
    )
    assert "--enforce-eager" not in command
    forbidden_fragments = (
        "prefill-context-parallel",
        "kv-transfer",
        "deepep_high_throughput",
        "deepep_low_latency",
    )
    assert not any(
        fragment in item for item in command for fragment in forbidden_fragments
    )
    assert "--moe-backend" not in command


def test_deepseek_v4_dspark_tp8_humaneval_command_contract() -> None:
    command, _, _ = server_command(_load("tp8"), model_env=MODEL_ENV)

    assert _option_value(command, "--tensor-parallel-size") == "8"
    assert "--data-parallel-size" not in command
    assert "--enable-expert-parallel" not in command
    assert "--all2all-backend" not in command


def test_deepseek_v4_dspark_dp8_humaneval_is_single_service_auto_deepep() -> None:
    command, _, _ = server_command(_load("dp8_ep8"), model_env=MODEL_ENV)

    assert _option_value(command, "--tensor-parallel-size") == "1"
    assert _option_value(command, "--data-parallel-size") == "8"
    assert "--enable-expert-parallel" in command
    assert _option_value(command, "--all2all-backend") == "deepep_auto"


@pytest.mark.parametrize("profile", ["tp8", "dp8_ep8"])
def test_deepseek_v4_dspark_humaneval_requires_exact_32_of_32(
    profile: str,
) -> None:
    config = _load(profile)

    assert config["evalscope"]["limit"] == 32
    assert config["evalscope"]["eval_batch_size"] == 1
    assert config["evalscope"]["pass_criteria"] == {
        "dataset": "humaneval",
        "num_predictions": 32,
        "num_reviews": 32,
        "normalize_code_fences": True,
        "mean_acc": 1.0,
        "mean_acc_pass@1": 1.0,
    }


@pytest.mark.parametrize("profile", ["tp8", "dp8_ep8"])
def test_deepseek_v4_dspark_humaneval_request_is_deterministic(
    tmp_path: Path,
    profile: str,
) -> None:
    config = _load(profile)
    command = evalscope_command(
        config,
        model_env=MODEL_ENV,
        host="127.0.0.1",
        port=10136,
        work_dir=tmp_path,
    )
    generation = json.loads(_option_value(command, "--generation-config"))

    assert generation == {
        "temperature": 0,
        "do_sample": False,
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
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
def test_deepseek_v4_dspark_humaneval_tp8() -> None:
    config = _load("tp8")
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="DeepSeek-V4-Flash-0731 TP8+DSpark",
        required_hcu_count=8,
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_deepseek_v4_dspark_humaneval_dp8_ep8() -> None:
    config = _load("dp8_ep8")
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="DeepSeek-V4-Flash-0731 DP8+EP8+DSpark",
        required_hcu_count=8,
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_deepseek_v4_int8_dspark_humaneval_tp8() -> None:
    config = _load_int8("tp8")
    run_evalscope_server_test(
        config,
        model_env=INT8_MODEL_ENV,
        model_label="DeepSeek-V4-Flash-0731 INT8 TP8+DSpark",
        required_hcu_count=8,
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_deepseek_v4_int8_dspark_humaneval_dp8_ep8() -> None:
    config = _load_int8("dp8_ep8")
    run_evalscope_server_test(
        config,
        model_env=INT8_MODEL_ENV,
        model_label="DeepSeek-V4-Flash-0731 INT8 DP8+EP8+DSpark",
        required_hcu_count=8,
    )
