# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4 FP8/INT8 DSpark Mooncake P/D acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server import pd_evalscope_server as pd_runner
from tests.integration.server.evalscope_server import load_profiled_config
from tests.integration.server.pd_evalscope_server import (
    assert_pd_runtime_evidence,
    pd_commands,
    run_evalscope_pd_server_test,
)


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
PREFILL_EVIDENCE = """\
Mooncake TTFT_EVENT event=p_send_kv_done ts=1.0
DeepEP auto selected contiguous high-throughput experts for this forward.
Using DeepEPDeepGemmContiguousExperts with DeepGEMM HT path.
"""
DECODE_EVIDENCE = """\
Mooncake TTFT_EVENT event=d_kv_ready ts=2.0
DeepEP auto selected masked low-latency experts for this forward.
Using DeepEPDeepGemmMaskedExperts with DeepGEMM LL path.
"""
DSPARK_METRICS = """\
vllm:spec_decode_num_draft_tokens_total{engine="0"} 128
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 64
"""


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


def test_deepseek_v4_dspark_mooncake_pd_runtime_evidence(
    tmp_path: Path,
) -> None:
    prefill_log = tmp_path / "prefill.log"
    decode_log = tmp_path / "decode.log"
    prefill_log.write_text(PREFILL_EVIDENCE, encoding="utf-8")
    decode_log.write_text(DECODE_EVIDENCE, encoding="utf-8")

    assert_pd_runtime_evidence(prefill_log, decode_log, DSPARK_METRICS)


@pytest.mark.parametrize(
    ("prefill_text", "decode_text", "metrics", "match"),
    [
        (
            PREFILL_EVIDENCE,
            DECODE_EVIDENCE,
            DSPARK_METRICS.replace(
                'spec_decode_num_draft_tokens_total{engine="0"} 128',
                'spec_decode_num_draft_tokens_total{engine="0"} 0',
            ),
            "draft tokens",
        ),
        (
            PREFILL_EVIDENCE,
            DECODE_EVIDENCE,
            DSPARK_METRICS.replace(
                'spec_decode_num_accepted_tokens_total{engine="0"} 64',
                'spec_decode_num_accepted_tokens_total{engine="0"} 0',
            ),
            "accepted tokens",
        ),
        (
            PREFILL_EVIDENCE,
            DECODE_EVIDENCE.replace(
                "Mooncake TTFT_EVENT event=d_kv_ready ts=2.0\n", ""
            ),
            DSPARK_METRICS,
            "d_kv_ready",
        ),
        (
            PREFILL_EVIDENCE + "Sending to 127.0.0.1 failed (ret=-1)\n",
            DECODE_EVIDENCE,
            DSPARK_METRICS,
            "Mooncake transfer failure",
        ),
        (
            PREFILL_EVIDENCE,
            DECODE_EVIDENCE
            + "MooncakeXferMetadata transfer failed for request: error\n",
            DSPARK_METRICS,
            "Mooncake transfer failure",
        ),
    ],
)
def test_deepseek_v4_dspark_mooncake_pd_rejects_missing_runtime_evidence(
    tmp_path: Path,
    prefill_text: str,
    decode_text: str,
    metrics: str,
    match: str,
) -> None:
    prefill_log = tmp_path / "prefill.log"
    decode_log = tmp_path / "decode.log"
    prefill_log.write_text(prefill_text, encoding="utf-8")
    decode_log.write_text(decode_text, encoding="utf-8")

    with pytest.raises(AssertionError, match=match):
        assert_pd_runtime_evidence(prefill_log, decode_log, metrics)


def test_deepseek_v4_dspark_mooncake_pd_process_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeProcess:
        def __init__(self, label: str) -> None:
            self.label = label
            self.pid = {"prefill": 101, "decode": 102, "proxy": 103}[label]

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        if "--port" in command:
            port = command[command.index("--port") + 1]
        else:
            raise AssertionError(f"command has no port: {command}")
        label = {"10141": "prefill", "10142": "decode", "10140": "proxy"}[
            port
        ]
        events.append(f"start:{label}")
        return FakeProcess(label)

    def fake_wait(proc: FakeProcess, _url: str, _timeout_s: int) -> None:
        events.append(f"wait:{proc.label}")

    def fake_smoke(
        proc: FakeProcess,
        _url: str,
        _model: str,
        _timeout_s: int,
    ) -> None:
        events.append(f"smoke:{proc.label}")

    def fake_run(command: list[str], **_kwargs: object) -> object:
        api_url = command[command.index("--api-url") + 1]
        assert api_url == "http://127.0.0.1:10140/v1"
        events.append("eval:10140")
        return type("Result", (), {"returncode": 0})()

    def fake_fetch(url: str, _timeout: int) -> str:
        assert url == "http://127.0.0.1:10142/metrics"
        events.append("metrics:decode")
        return DSPARK_METRICS

    def fake_stop(proc: FakeProcess, _timeout_s: int) -> None:
        events.append(f"stop:{proc.label}")

    monkeypatch.setenv("VLLM_V0251_SOURCE_ROOT", "/models/zb/vllm_025/vllm")
    monkeypatch.setenv("VLLM_HCU_EVAL_WORK_DIR", str(tmp_path / "eval"))
    monkeypatch.setattr(pd_runner, "_require_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pd_runner, "_reset_evalscope_artifacts", lambda _work_dir: None
    )
    monkeypatch.setattr(pd_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pd_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(pd_runner, "_wait_for_server", fake_wait)
    monkeypatch.setattr(pd_runner, "_wait_for_routed_smoke", fake_smoke)
    monkeypatch.setattr(pd_runner, "_fetch_text", fake_fetch)
    monkeypatch.setattr(
        pd_runner,
        "_assert_pass_criteria",
        lambda *args, **kwargs: events.append("accuracy"),
    )
    monkeypatch.setattr(
        pd_runner,
        "assert_pd_runtime_evidence",
        lambda *args, **kwargs: events.append("evidence"),
    )
    monkeypatch.setattr(pd_runner, "_terminate_process_group", fake_stop)

    run_evalscope_pd_server_test(
        _load("fp8"),
        model_env=MODEL_ENV,
        model_label="DeepSeek V4 FP8",
        required_hcu_count=8,
    )

    assert events == [
        "start:prefill",
        "wait:prefill",
        "start:decode",
        "wait:decode",
        "start:proxy",
        "smoke:proxy",
        "eval:10140",
        "metrics:decode",
        "accuracy",
        "evidence",
        "stop:proxy",
        "stop:decode",
        "stop:prefill",
    ]
