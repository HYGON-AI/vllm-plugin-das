# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Primary-model HumanEval-32 acceptance for adapted HCU operators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    _report_metric,
    _server_environment,
    load_profiled_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEEPSEEK_CONFIG = ROOT / "tests/models/deepseek_v4_int8_humaneval_evalscope.yaml"
QWEN_CONFIG = ROOT / "tests/models/qwen36_35b_a3b_humaneval_evalscope.yaml"
QWEN_27B_CONFIG = ROOT / "tests/models/qwen36_27b_humaneval_evalscope.yaml"
QWEN_35_W8A8_CONFIG = (
    ROOT / "tests/models/qwen35_35b_a3b_w8a8_humaneval_evalscope.yaml"
)
DEEPSEEK_CONFIG_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_DEEPSEEK_CONFIG"
QWEN_CONFIG_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_CONFIG"
DEEPSEEK_MODEL_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_DEEPSEEK_MODEL"
QWEN_MODEL_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_MODEL"
QWEN_27B_MODEL_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_27B_MODEL"
QWEN_35_W8A8_MODEL_ENV = "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_35_W8A8_MODEL"


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.parametrize(
    ("config_path", "config_env", "model_env", "model_path", "tp", "leaf"),
    [
        (
            DEEPSEEK_CONFIG,
            DEEPSEEK_CONFIG_ENV,
            DEEPSEEK_MODEL_ENV,
            "/models/DeepSeek-V4-Flash-Channel-INT8-w8a8",
            "4",
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        ),
        (
            QWEN_CONFIG,
            QWEN_CONFIG_ENV,
            QWEN_MODEL_ENV,
            "/models/Qwen3.6-35B-A3B",
            "1",
            (
                "VLLM_HCU_USE_LIGHTOP_W16A16_MOE",
                "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED",
            ),
        ),
        (
            QWEN_27B_CONFIG,
            "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_27B_CONFIG",
            QWEN_27B_MODEL_ENV,
            "/models/Qwen3.6-27B",
            "1",
            "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED",
        ),
        (
            QWEN_35_W8A8_CONFIG,
            "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_35_W8A8_CONFIG",
            QWEN_35_W8A8_MODEL_ENV,
            "/models/Qwen3.5-35B-A3B-W8A8",
            "1",
            "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED",
        ),
    ],
)
@pytest.mark.parametrize(("profile", "enabled"), [("feature_off", "0"), ("feature_on", "1")])
def test_operator_adaptation_humaneval_config_contract(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    config_env: str,
    model_env: str,
    model_path: str,
    tp: str,
    leaf: str | tuple[str, ...],
    profile: str,
    enabled: str,
) -> None:
    monkeypatch.delenv(config_env, raising=False)
    monkeypatch.delenv(model_env, raising=False)
    config = load_profiled_config(config_path, config_env, profile=profile)
    command, host, port = server_command(config, model_env=model_env)
    environment = _server_environment(config)
    evaluation = config["evalscope"]

    assert command[:3] == ["vllm", "serve", model_path]
    assert host == "127.0.0.1"
    assert port > 0
    assert _option_value(command, "--tensor-parallel-size") == tp
    assert environment["VLLM_HCU_USE_CUSTOM_OPS"] == "1"
    leaves = (leaf,) if isinstance(leaf, str) else leaf
    assert all(environment[name] == enabled for name in leaves)
    assert evaluation["limit"] == 32
    assert evaluation["eval_batch_size"] == 1
    assert evaluation["datasets"] == ["humaneval"]
    assert evaluation["generation_config"]["temperature"] == 0
    assert evaluation["generation_config"]["do_sample"] is False
    assert evaluation["pass_criteria"] == {
        "dataset": "humaneval",
        "metric": "mean_acc_pass@1",
        "display_name": "Pass@1",
        "minimum_score": 0.80,
        "num_predictions": 32,
        "num_reviews": 32,
    }

    work_dir = Path(evaluation["work_dir"])
    assert work_dir.parent == Path("/tmp/vllm-hcu-evalscope")
    assert profile.replace("feature_", "") in work_dir.name


def _run_and_read(
    config: dict,
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
) -> tuple[float, float, Path, Path, str]:
    work_dir = Path(config["evalscope"]["work_dir"])
    server_log = work_dir / "logs/vllm_server.log"
    log_offset = server_log.stat().st_size if server_log.exists() else 0
    run_evalscope_server_test(
        config,
        model_env=model_env,
        model_label=model_label,
        required_hcu_count=required_hcu_count,
    )
    served_model = str(config["server"]["served_model_name"])
    score, samples, report_path = _report_metric(
        work_dir,
        model=served_model,
        dataset="humaneval",
        metric="mean_acc_pass@1",
    )
    assert samples == 32
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_tps = float(report["perf_metrics"]["summary"]["throughput"]["avg_output_tps"])
    assert output_tps > 0
    with server_log.open("rb") as stream:
        stream.seek(log_offset)
        invocation_log = stream.read().decode("utf-8", errors="replace")
    return score, output_tps, report_path, server_log, invocation_log


def _assert_feature_pair(
    config_path: Path,
    config_env: str,
    *,
    model_env: str,
    model_label: str,
    required_hcu_count: int,
    route_messages: str | tuple[str, ...],
    expect_feature_on_route: bool = True,
) -> None:
    results = {}
    for profile in ("feature_off", "feature_on"):
        config = load_profiled_config(config_path, config_env, profile=profile)
        results[profile] = _run_and_read(
            config,
            model_env=model_env,
            model_label=f"{model_label} {profile}",
            required_hcu_count=required_hcu_count,
        )
    off_score, off_tps, off_report, off_log, off_invocation = results["feature_off"]
    on_score, on_tps, on_report, on_log, on_invocation = results["feature_on"]
    expected_messages = (
        (route_messages,) if isinstance(route_messages, str) else route_messages
    )
    for route_message in expected_messages:
        assert route_message not in off_invocation
        if expect_feature_on_route:
            assert route_message in on_invocation, (
                f"feature-on did not execute the requested HCU route; log={on_log}"
            )
        else:
            assert route_message not in on_invocation, (
                f"ineligible model unexpectedly executed HCU route; log={on_log}"
            )
    assert on_score >= off_score, (
        f"feature-on Pass@1 {on_score:.4f} regressed below feature-off "
        f"{off_score:.4f}; reports={off_report},{on_report}"
    )
    print(
        "operator-adaptation throughput observation: "
        f"off={off_tps:.2f} tok/s, on={on_tps:.2f} tok/s, "
        f"reports={off_report},{on_report}"
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(4)
@pytest.mark.slow
@pytest.mark.external_service("evalscope")
def test_deepseek_v4_int8_operator_adaptation_humaneval32() -> None:
    _assert_feature_pair(
        DEEPSEEK_CONFIG,
        DEEPSEEK_CONFIG_ENV,
        model_env=DEEPSEEK_MODEL_ENV,
        model_label="DeepSeek-V4-Flash INT8 TP4",
        required_hcu_count=4,
        route_messages="Using LightOp sqrt-softplus MoE routing.",
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.hcu_count(1)
@pytest.mark.slow
@pytest.mark.external_service("evalscope")
def test_qwen36_35b_a3b_operator_adaptation_humaneval32() -> None:
    _assert_feature_pair(
        QWEN_CONFIG,
        QWEN_CONFIG_ENV,
        model_env=QWEN_MODEL_ENV,
        model_label="Qwen3.6-35B-A3B TP1",
        required_hcu_count=1,
        route_messages=(
            "Using LightOp W16A16 Marlin MoE backend.",
            "Using LightOp Qwen gated RMSNorm.",
        ),
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.hcu_count(1)
@pytest.mark.slow
@pytest.mark.external_service("evalscope")
def test_qwen36_27b_gated_rmsnorm_humaneval32() -> None:
    _assert_feature_pair(
        QWEN_27B_CONFIG,
        "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_27B_CONFIG",
        model_env=QWEN_27B_MODEL_ENV,
        model_label="Qwen3.6-27B TP1",
        required_hcu_count=1,
        route_messages="Using LightOp Qwen gated RMSNorm.",
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.hcu_count(1)
@pytest.mark.slow
@pytest.mark.external_service("evalscope")
def test_qwen35_35b_a3b_w8a8_gated_rmsnorm_fallback_humaneval32() -> None:
    _assert_feature_pair(
        QWEN_35_W8A8_CONFIG,
        "VLLM_HCU_OPERATOR_ADAPTATION_QWEN_35_W8A8_CONFIG",
        model_env=QWEN_35_W8A8_MODEL_ENV,
        model_label="Qwen3.5-35B-A3B-W8A8 TP1",
        required_hcu_count=1,
        route_messages="Using LightOp Qwen gated RMSNorm.",
        expect_feature_on_route=False,
    )
