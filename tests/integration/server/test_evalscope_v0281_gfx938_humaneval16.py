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
    load_config,
    load_profiled_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/models/v0281_gfx938_humaneval16.yaml"
CONFIG_ENV = "VLLM_HCU_GFX938_HUMANEVAL16_CONFIG"
PROFILE_ENV = "VLLM_HCU_GFX938_PROFILE"
MODEL_ENV = "VLLM_HCU_GFX938_MODEL"


def _required_hcu_count(profile: str, tensor_parallel_size: int) -> int:
    if profile == "hy3_channel_fp8_dp8_ep8_mtp2_kvfp8":
        return 8
    return tensor_parallel_size


PROFILE_CONTRACTS = (
    (
        "deepseek_v32_channel_fp8_tp8",
        "/models/DeepSeek-V3.2-channel-fp8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "deepseek_v32_channel_fp8_mtp3_kvfp8_tp8",
        "/models/DeepSeek-V3.2-channel-fp8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "deepseek_r1_channel_fp8_tp8",
        "/models/DeepSeek-R1-Channel-FP8-w8a8",
        8,
        "FLASHMLA",
        None,
    ),
    (
        "deepseek_r1_channel_fp8_mtp3_tp8",
        "/models/DeepSeek-R1-Channel-FP8-w8a8",
        8,
        "FLASHMLA",
        None,
    ),
    (
        "deepseek_r1_0528_channel_int8_kvfp8_tp8",
        "/models/DeepSeek-R1-0528-Channel-INT8",
        8,
        "FLASHMLA",
        None,
    ),
    (
        "glm52_channel_int8_tp8",
        "/models/GLM-5.2-Channel-INT8-w8a8",
        8,
        "FLASHMLA_SPARSE",
        None,
    ),
    (
        "glm47_w8a8_mtp2_kvfp8_tp4",
        "/models/GLM-4.7-W8A8",
        4,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "deepseek_v4_flash_tp8",
        "/models/DeepSeek-V4-Flash-0731-FP8-Channel",
        8,
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
        "hy3_channel_fp8_mtp2_kvfp8_tp8",
        "/models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3",
        8,
        "FLASH_ATTN",
        "HND",
    ),
    (
        "hy3_channel_fp8_dp8_ep8_mtp2_kvfp8",
        "/models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3",
        1,
        "FLASH_ATTN",
        "HND",
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
        None,
    ),
    (
        "qwen38_flash_next_w4a8_tp4",
        "/models/Qwen3.8-Flash-Next-w4a8-slimquant",
        4,
        "FLASH_ATTN",
        None,
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
    expected_eval_batch_size = "1" if profile == "deepseek_v4_flash_tp8" else "8"
    assert (
        _option_value(evaluation, "--eval-batch-size")
        == expected_eval_batch_size
    )
    assert config["evalscope"]["generation_config"]["temperature"] == 0
    diagnostic_profiles = {
        "deepseek_r1_0528_channel_int8_kvfp8_tp8",
        "deepseek_v4_flash_tp8",
        "qwen2_57b_tp2",
        "qwen3_30b_int8_tp2",
        "qwen36_27b_w8a8_tp2",
    }
    assert bool(
        config["evalscope"]["pass_criteria"].get("enforce_score", True)
    ) is (profile not in diagnostic_profiles)
    assert config["evalscope"]["generation_config"]["do_sample"] is False
    output_token_overrides = {
        "deepseek_v32_channel_fp8_tp8": 4096,
        "deepseek_v32_channel_fp8_mtp3_kvfp8_tp8": 4096,
        "deepseek_r1_channel_fp8_tp8": 4096,
        "deepseek_r1_channel_fp8_mtp3_tp8": 4096,
        "deepseek_r1_0528_channel_int8_kvfp8_tp8": 8192,
    }
    expected_max_tokens = output_token_overrides.get(profile, 2048)
    assert (
        config["evalscope"]["generation_config"]["max_tokens"]
        == expected_max_tokens
    )
    expected_criteria = {
        "dataset": "humaneval",
        "mean_acc": 1.0,
        "mean_acc_pass@1": 1.0,
        "num_predictions": 16,
        "num_reviews": 16,
        "normalize_code_fences": True,
    }
    if profile in diagnostic_profiles:
        expected_criteria["enforce_score"] = False
    assert config["evalscope"]["pass_criteria"] == expected_criteria
    assert config["server"]["prefix_probe"]["metric"] == (
        "vllm:prefix_cache_hits_total"
    )
    expected_content_repeat = (
        64
        if profile
        in {"qwen35_35b_w8a8_tp2", "qwen38_flash_next_fp8_tp4"}
        else 32
    )
    assert (
        config["server"]["prefix_probe"]["content_repeat"]
        == expected_content_repeat
    )
    environment = config["server"].get("environment", {})
    assert environment.get("VLLM_KV_CACHE_LAYOUT") == kv_layout
    if attention != "FLASH_ATTN":
        assert "VLLM_KV_CACHE_LAYOUT" not in environment


def test_gfx938_profiles_have_unique_ports_and_owned_work_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    expected_profiles = {profile for profile, *_ in PROFILE_CONTRACTS}
    raw_config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    assert set(raw_config["profiles"]) == expected_profiles

    work_dirs = []
    ports = []
    for profile, *_ in PROFILE_CONTRACTS:
        config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
        work_dir = str(config["evalscope"]["work_dir"])
        assert work_dir.startswith("/tmp/vllm-hcu-evalscope/")
        work_dirs.append(work_dir)
        ports.append(config["server"]["port"])
    assert len(work_dirs) == len(set(work_dirs)) == 24
    assert len(ports) == len(set(ports)) == 24


def test_gfx938_profile_specific_reasoning_and_mtp_contracts() -> None:
    glm47 = load_profiled_config(
        DEFAULT_CONFIG,
        CONFIG_ENV,
        profile="glm47_w8a8_mtp2_kvfp8_tp4",
    )
    glm47_command, _, _ = server_command(glm47, model_env=MODEL_ENV)
    assert glm47["server"]["environment"] == {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_KV_CACHE_LAYOUT": "HND",
    }
    assert json.loads(
        _option_value(glm47_command, "--speculative-config")
    ) == {"method": "mtp", "num_speculative_tokens": 2}
    assert _option_value(glm47_command, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(glm47_command, "--moe-backend") == "aiter"
    assert _option_value(glm47_command, "--reasoning-parser") == "glm47"
    assert json.loads(
        _option_value(glm47_command, "--default-chat-template-kwargs")
    ) == {"enable_thinking": False}
    assert glm47["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    glm52 = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="glm52_channel_int8_tp8"
    )
    glm52_command, _, _ = server_command(glm52, model_env=MODEL_ENV)
    assert glm52["server"]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert json.loads(
        _option_value(glm52_command, "--speculative-config")
    ) == {"method": "mtp", "num_speculative_tokens": 3}
    assert _option_value(glm52_command, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(glm52_command, "--moe-backend") == "aiter"
    assert _option_value(glm52_command, "--reasoning-parser") == "glm45"
    assert json.loads(
        _option_value(glm52_command, "--default-chat-template-kwargs")
    ) == {"enable_thinking": False}
    assert glm52["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    v32_commands = {}
    for profile in (
        "deepseek_v32_channel_fp8_tp8",
        "deepseek_v32_channel_fp8_mtp3_kvfp8_tp8",
    ):
        config = load_profiled_config(
            DEFAULT_CONFIG, CONFIG_ENV, profile=profile
        )
        command, _, _ = server_command(config, model_env=MODEL_ENV)
        assert config["server"]["environment"][
            "VLLM_USE_V2_MODEL_RUNNER"
        ] == "1"
        assert _option_value(command, "--max-model-len") == "8192"
        assert _option_value(command, "--max-num-batched-tokens") == "4096"
        assert json.loads(
            _option_value(command, "--default-chat-template-kwargs")
        ) == {"thinking": False}
        assert config["evalscope"]["generation_config"]["extra_body"] == {
            "chat_template_kwargs": {"thinking": False}
        }
        v32_commands[profile] = command

    assert "--speculative-config" not in v32_commands[
        "deepseek_v32_channel_fp8_tp8"
    ]
    assert "--kv-cache-dtype" not in v32_commands[
        "deepseek_v32_channel_fp8_tp8"
    ]
    assert json.loads(
        _option_value(
            v32_commands["deepseek_v32_channel_fp8_mtp3_kvfp8_tp8"],
            "--speculative-config",
        )
    ) == {"method": "mtp", "num_speculative_tokens": 3}
    assert _option_value(
        v32_commands["deepseek_v32_channel_fp8_mtp3_kvfp8_tp8"],
        "--kv-cache-dtype",
    ) == "fp8_e4m3"

    r1_commands = {}
    for profile in (
        "deepseek_r1_channel_fp8_tp8",
        "deepseek_r1_channel_fp8_mtp3_tp8",
        "deepseek_r1_0528_channel_int8_kvfp8_tp8",
    ):
        config = load_profiled_config(
            DEFAULT_CONFIG, CONFIG_ENV, profile=profile
        )
        command, _, _ = server_command(config, model_env=MODEL_ENV)
        assert config["server"]["environment"][
            "VLLM_USE_V2_MODEL_RUNNER"
        ] == "1"
        assert _option_value(command, "--reasoning-parser") == "deepseek_r1"
        assert _option_value(command, "--max-model-len") == "32768"
        assert _option_value(command, "--max-num-batched-tokens") == "16384"
        r1_commands[profile] = command

    assert "--speculative-config" not in r1_commands[
        "deepseek_r1_channel_fp8_tp8"
    ]
    assert json.loads(
        _option_value(
            r1_commands["deepseek_r1_channel_fp8_mtp3_tp8"],
            "--speculative-config",
        )
    ) == {"method": "mtp", "num_speculative_tokens": 3}
    r1_0528_command = r1_commands[
        "deepseek_r1_0528_channel_int8_kvfp8_tp8"
    ]
    assert "--speculative-config" not in r1_0528_command
    assert _option_value(r1_0528_command, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(r1_0528_command, "--moe-backend") == "aiter"

    deepseek_v4 = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="deepseek_v4_flash_tp8"
    )
    deepseek_v4_command, _, _ = server_command(deepseek_v4, model_env=MODEL_ENV)
    assert deepseek_v4["server"]["environment"][
        "VLLM_USE_V2_MODEL_RUNNER"
    ] == "1"
    assert json.loads(
        _option_value(deepseek_v4_command, "--speculative-config")
    ) == {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    }
    assert deepseek_v4["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"thinking": False}
    }

    hy4 = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="hy4_preview_channel_fp8_tp8"
    )
    hy4_command, _, _ = server_command(hy4, model_env=MODEL_ENV)
    assert json.loads(
        _option_value(hy4_command, "--default-chat-template-kwargs")
    ) == {"reasoning_effort": "no_think"}
    assert hy4["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "no_think"}
    }

    hy3 = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="hy3_channel_fp8_mtp2_kvfp8_tp8"
    )
    hy3_command, _, _ = server_command(hy3, model_env=MODEL_ENV)
    assert hy3["server"]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert json.loads(
        _option_value(hy3_command, "--speculative-config")
    ) == {"method": "mtp", "num_speculative_tokens": 2}
    assert _option_value(hy3_command, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(hy3_command, "--moe-backend") == "aiter"
    assert _option_value(hy3_command, "--reasoning-parser") == "hy_v3"
    assert json.loads(
        _option_value(hy3_command, "--default-chat-template-kwargs")
    ) == {"reasoning_effort": "no_think"}
    assert hy3["evalscope"]["generation_config"]["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "no_think"}
    }

    minimax = load_profiled_config(
        DEFAULT_CONFIG, CONFIG_ENV, profile="minimax_m25_int8_tp4"
    )
    minimax_command, _, _ = server_command(minimax, model_env=MODEL_ENV)
    assert _option_value(minimax_command, "--reasoning-parser") == (
        "minimax_m2_append_think"
    )
    assert "--generation-config" not in minimax_command
    assert "--speculative-config" not in minimax_command

    for profile in (
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

    for profile in ("qwen35_35b_w8a8_tp2", "qwen38_flash_next_fp8_tp4"):
        fp8_kv = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
        fp8_kv_command, _, _ = server_command(fp8_kv, model_env=MODEL_ENV)
        assert _option_value(fp8_kv_command, "--max-model-len") == "8192"
        assert fp8_kv["server"]["prefix_probe"]["content_repeat"] == 64


def test_hy3_dp8_ep8_low_latency_contract() -> None:
    config = load_profiled_config(
        DEFAULT_CONFIG,
        CONFIG_ENV,
        profile="hy3_channel_fp8_dp8_ep8_mtp2_kvfp8",
    )
    command, _, _ = server_command(config, model_env=MODEL_ENV)

    assert _option_value(command, "--tensor-parallel-size") == "1"
    assert _option_value(command, "--data-parallel-size") == "8"
    assert config["server"]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert "--enable-expert-parallel" in command
    assert _option_value(command, "--all2all-backend") == "deepep_low_latency"
    assert _option_value(command, "--moe-backend") == "deep_gemm"
    assert json.loads(
        _option_value(command, "--speculative-config")
    ) == {"method": "mtp", "num_speculative_tokens": 2}
    assert _option_value(command, "--kv-cache-dtype") == "fp8_e4m3"
    assert config["server"]["prefix_probe"]["request_count"] == 9
    assert "--enforce-eager" not in command
    assert "--compilation-config" not in command


def test_hy3_dp8_requires_all_eight_hcus_despite_tp1() -> None:
    assert _required_hcu_count("hy3_channel_fp8_dp8_ep8_mtp2_kvfp8", 1) == 8
    assert _required_hcu_count("hy3_channel_fp8_mtp2_kvfp8_tp8", 8) == 8


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
    is_hy3_dp8 = profile == "hy3_channel_fp8_dp8_ep8_mtp2_kvfp8"
    required_hcu_count = _required_hcu_count(profile, tp)
    topology_label = "DP8/TP1/EP8" if is_hy3_dp8 else f"TP{tp}"
    config = load_profiled_config(DEFAULT_CONFIG, CONFIG_ENV, profile=profile)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label=f"{Path(model).name} gfx938 {topology_label}",
        required_hcu_count=required_hcu_count,
    )
