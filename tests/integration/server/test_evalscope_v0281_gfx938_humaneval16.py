# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""gfx938 vLLM 0.28.1 TP and HumanEval16 acceptance matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    evalscope_command,
    load_profiled_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/models/v0281_gfx938_humaneval16.yaml"
CONFIG_ENV = "VLLM_HCU_GFX938_HUMANEVAL16_CONFIG"
PROFILE_ENV = "VLLM_HCU_GFX938_PROFILE"
MODEL_ENV = "VLLM_HCU_GFX938_MODEL"

PROFILE_CONTRACTS = (
    (
        "deepseek_v4_flash_tp4",
        "/models/DeepSeek-V4-Flash-0731-FP8-Channel",
        4,
        "FLASHMLA_SPARSE",
        None,
    ),
    ("glm5_w8a8_tp8", "/models/GLM-5-W8A8", 8, "FLASHMLA_SPARSE", None),
    (
        "glm53_channel_fp8_tp8",
        "/models/GLM-5.3-Channel-FP8-w8a8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "glm51_channel_fp8_tp8",
        "/models/GLM-5___1-Channel-FP8-w8a8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "hy4_preview_channel_fp8_tp8",
        "/models/Hy4-preview-Channel-FP8-w8a8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "minimax_m25_int8_tp4",
        "/models/MiniMax-M2.5-Channel-INT8-w8a8",
        4,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen2_57b_tp2",
        "/models/Qwen2-57B-A14B-Instruct",
        2,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen3_30b_int8_tp2",
        "/models/Qwen3-30B-A3B-Channel-INT8-w8a8",
        2,
        "FLASH_ATTN",
        "HND",
    ),
    ("qwen3_8b_tp2", "/models/Qwen3-8B", 2, "FLASH_ATTN", "HND"),
    ("qwen35_35b_tp2", "/models/Qwen3.5-35B-A3B", 2, "FLASH_ATTN", "HND"),
    (
        "qwen35_35b_w8a8_tp2",
        "/models/Qwen3.5-35B-A3B-W8A8",
        2,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen36_27b_w8a8_tp2",
        "/models/Qwen3.6-27B-W8A8",
        2,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen38_27b_int8_tp2",
        "/models/Qwen3.8-27B-Channel-INT8-w8a8",
        2,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen38_flash_next_fp8_tp4",
        "/models/Qwen3.8-Flash-Next-FP8-Channelwise",
        4,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "qwen38_flash_next_w4a8_tp4",
        "/models/Qwen3.8-Flash-Next-w4a8-slimquant",
        4,
        "FLASH_ATTN",
        "HND",
    ),
)


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.parametrize(
    ("profile", "model", "tp", "attention", "kv_layout"),
    PROFILE_CONTRACTS,
)
def test_gfx938_profile_contract(
    profile: str,
    model: str,
    tp: int,
    attention: str,
    kv_layout: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
    server, host, port = server_command(config, model_env=MODEL_ENV)
    evaluation = evalscope_command(
        config,
        model_env=MODEL_ENV,
        host=host,
        port=port,
        work_dir=tmp_path,
    )

    assert server[:3] == ["vllm", "serve", model]
    assert _option_value(server, "--tensor-parallel-size") == str(tp)
    assert _option_value(server, "--attention-backend") == attention
    assert "--enable-prefix-caching" in server
    assert "--enforce-eager" not in server
    assert "--compilation-config" not in server
    assert _option_value(evaluation, "--datasets") == "humaneval"
    assert _option_value(evaluation, "--limit") == "16"
    assert config["evalscope"]["generation_config"]["temperature"] == 0
    assert config["evalscope"]["generation_config"]["do_sample"] is False
    assert config["evalscope"]["pass_criteria"] == {
        "dataset": "humaneval",
        "mean_acc": 1.0,
        "mean_acc_pass@1": 1.0,
        "num_predictions": 16,
        "num_reviews": 16,
        "normalize_code_fences": True,
    }
    assert config["server"]["prefix_probe"]["metric"] == (
        "vllm:prefix_cache_hits_total"
    )
    environment = config["server"].get("environment", {})
    assert environment.get("VLLM_KV_CACHE_LAYOUT") == kv_layout
    if attention != "FLASH_ATTN":
        assert "VLLM_KV_CACHE_LAYOUT" not in environment


def test_gfx938_profiles_have_unique_owned_work_directories() -> None:
    work_dirs = []
    for profile, *_ in PROFILE_CONTRACTS:
        config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
        work_dir = str(config["evalscope"]["work_dir"])
        assert work_dir.startswith("/tmp/vllm-hcu-evalscope/")
        work_dirs.append(work_dir)
    assert len(work_dirs) == len(set(work_dirs)) == 15


def test_gfx938_profile_specific_reasoning_and_mtp_contracts() -> None:
    minimax = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="minimax_m25_int8_tp4"
    )
    minimax_command, _, _ = server_command(minimax, model_env=MODEL_ENV)
    assert _option_value(minimax_command, "--reasoning-parser") == "minimax_m2"
    assert "--speculative-config" not in minimax_command

    for profile in (
        "deepseek_v4_flash_tp4",
        "glm5_w8a8_tp8",
        "glm53_channel_fp8_tp8",
        "glm51_channel_fp8_tp8",
        "hy4_preview_channel_fp8_tp8",
        "qwen35_35b_tp2",
        "qwen35_35b_w8a8_tp2",
        "qwen38_flash_next_fp8_tp4",
        "qwen38_flash_next_w4a8_tp4",
    ):
        config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
        command, _, _ = server_command(config, model_env=MODEL_ENV)
        assert json.loads(_option_value(command, "--speculative-config")) == {
            "method": "mtp",
            "num_speculative_tokens": 3,
        }


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_v0281_gfx938_selected_profile_humaneval16() -> None:
    profile = os.environ.get(PROFILE_ENV)
    if not profile:
        pytest.skip(f"set {PROFILE_ENV} to one named gfx938 profile")
    contracts = {row[0]: row for row in PROFILE_CONTRACTS}
    if profile not in contracts:
        pytest.fail(f"unknown {PROFILE_ENV}={profile!r}")
    _, model, tp, _, _ = contracts[profile]
    config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label=f"{Path(model).name} gfx938 TP{tp}",
        required_hcu_count=tp,
    )
