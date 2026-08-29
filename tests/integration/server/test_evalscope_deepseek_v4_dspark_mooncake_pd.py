# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4 FP8/INT8 DSpark Mooncake P/D acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import load_profiled_config
from tests.integration.server.pd_evalscope_server import pd_commands


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "tests/models/deepseek_v4_flash_0731_dspark_mooncake_pd_humaneval.yaml"
)
CONFIG_ENV = "VLLM_HCU_DEEPSEEK_V4_DSPARK_MOONCAKE_PD_CONFIG"
MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_DSPARK_MOONCAKE_PD_MODEL"
DSPARK_CONFIG = {
    "method": "dspark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
}


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _load(profile: str) -> dict:
    return load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)


@pytest.mark.parametrize(
    ("profile", "model_path", "served_model_name"),
    [
        (
            "fp8",
            "/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
            "DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
        ),
        (
            "int8",
            "/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8",
            "DeepSeek-V4-Flash-0731-Channel-INT8-w8a8",
        ),
    ],
)
def test_deepseek_v4_dspark_mooncake_pd_command_contract(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    model_path: str,
    served_model_name: str,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    commands = pd_commands(_load(profile), model_env=MODEL_ENV)

    assert commands.prefill[:3] == ["vllm", "serve", model_path]
    assert commands.decode[:3] == ["vllm", "serve", model_path]
    assert commands.prefill_env["HIP_VISIBLE_DEVICES"] == "0,1,2,3"
    assert commands.decode_env["HIP_VISIBLE_DEVICES"] == "4,5,6,7"
    assert commands.prefill_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "18998"
    assert commands.decode_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "18999"
    assert commands.prefill_env["VLLM_DP_MASTER_PORT"] == "29561"
    assert commands.decode_env["VLLM_DP_MASTER_PORT"] == "29562"

    for command in (commands.prefill, commands.decode):
        assert _option_value(command, "--served-model-name") == served_model_name
        assert _option_value(command, "--data-parallel-size") == "4"
        assert _option_value(command, "--tensor-parallel-size") == "1"
        assert "--enable-expert-parallel" in command
        assert _option_value(command, "--all2all-backend") == "deepep_auto"
        assert _option_value(command, "--kv-cache-dtype") == "fp8"
        assert json.loads(_option_value(command, "--speculative-config")) == (
            DSPARK_CONFIG
        )
        assert "--moe-backend" not in command
        assert not any(
            fragment in item
            for item in command
            for fragment in (
                "prefill-context-parallel",
                "deepep_high_throughput",
                "deepep_low_latency",
            )
        )

    prefill_kv = json.loads(
        _option_value(commands.prefill, "--kv-transfer-config")
    )
    decode_kv = json.loads(_option_value(commands.decode, "--kv-transfer-config"))
    assert prefill_kv == {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
    }
    assert decode_kv == {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
    }
    assert commands.proxy[-9:] == [
        "--prefill",
        "http://127.0.0.1:10141",
        "18998",
        "--decode",
        "http://127.0.0.1:10142",
        "--host",
        "127.0.0.1",
        "--port",
        "10140",
    ]


@pytest.mark.parametrize("profile", ["fp8", "int8"])
def test_deepseek_v4_dspark_mooncake_pd_requires_exact_humaneval_32(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
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
