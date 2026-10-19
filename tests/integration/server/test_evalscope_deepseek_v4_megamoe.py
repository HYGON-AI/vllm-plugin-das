# SPDX-License-Identifier: Apache-2.0
"""Reusable HumanEval gate for HCU MegaMoE target and DSpark draft."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    _server_environment,
    load_profiled_config,
    run_evalscope_server_test,
    server_command,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/models/deepseek_v4_flash_megamoe_dspark_humaneval.yaml"
CONFIG_ENV = "VLLM_HCU_DEEPSEEK_V4_MEGAMOE_CONFIG"
MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL"
PROFILES = ("fp8_dspark", "w4a8_dspark", "fp8_dspark_graph")


@pytest.mark.parametrize("profile", PROFILES)
def test_megamoe_acceptance_configuration(profile):
    config = load_profiled_config(CONFIG, config_env=CONFIG_ENV, profile=profile)
    command, _, _ = server_command(config, model_env=MODEL_ENV)
    assert "--enable-expert-parallel" in command
    assert ("--enforce-eager" in command) == (profile != "fp8_dspark_graph")
    assert command[command.index("--moe-backend")+1] == "deep_gemm_mega_moe"
    assert command[command.index("--tensor-parallel-size")+1] == "8"
    assert "--all2all-backend" not in command
    spec = json.loads(command[command.index("--speculative-config")+1])
    assert spec["method"] == "dspark" and spec["num_speculative_tokens"] == 7
    if profile == "w4a8_dspark":
        assert spec["quantization"] == "slimquant_w4a8"
        assert "model" in spec
    env = _server_environment(config)
    assert env["VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"] == "0"


@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
@pytest.mark.parametrize("profile", PROFILES)
def test_megamoe_dspark_humaneval(profile):
    config = load_profiled_config(CONFIG, config_env=CONFIG_ENV, profile=profile)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label=f"DeepSeek-V4 Flash HCU MegaMoE {profile}",
        required_hcu_count=8,
    )
