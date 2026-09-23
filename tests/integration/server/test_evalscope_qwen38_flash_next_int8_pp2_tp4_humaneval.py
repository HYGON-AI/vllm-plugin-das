# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Qwen3.8 Flash Next INT8 PP2+TP4 OpenAI/HumanEval acceptance."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import (
    _server_environment,
    evalscope_command,
    load_config,
    run_evalscope_server_test,
    server_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "tests/models/qwen38_flash_next_int8_pp2_tp4_humaneval_evalscope.yaml"
)
CONFIG_ENV = "VLLM_HCU_QWEN38_FLASH_NEXT_INT8_PP2_TP4_HUMANEVAL_CONFIG"
MODEL_ENV = "VLLM_HCU_QWEN38_FLASH_NEXT_INT8_MODEL"


def test_qwen38_flash_next_int8_pp2_tp4_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)

    server, host, port = server_command(config, model_env=MODEL_ENV)
    environment = _server_environment(config)
    evaluation = evalscope_command(
        config,
        model_env=MODEL_ENV,
        host=host,
        port=port,
        work_dir=tmp_path,
    )

    model = "/models/Qwen3.8-Flash-Next-Channel-INT8-w8a8-ngram"
    assert server == [
        "vllm",
        "serve",
        model,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "2",
        "--moe-backend",
        "aiter",
        "--engram-config.cpu_offload=true",
        "--speculative-config.method",
        "mtp",
        "--speculative-config.num_speculative_tokens",
        "3",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--served-model-name",
        model.rsplit("/", 1)[-1],
        "--port",
        "10134",
    ]
    assert {
        key: environment[key]
        for key in (
            "HIP_VISIBLE_DEVICES",
            "VLLM_HCU_USE_AITER_MOE_SHUFFLE",
            "VLLM_HCU_PLE_PREFETCH_STREAM",
        )
    } == {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "VLLM_HCU_USE_AITER_MOE_SHUFFLE": "0",
        "VLLM_HCU_PLE_PREFETCH_STREAM": "1",
    }
    assert evaluation == [
        "evalscope",
        "eval",
        "--model",
        model.rsplit("/", 1)[-1],
        "--api-url",
        "http://127.0.0.1:10134/v1",
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--generation-config",
        (
            '{"temperature":0,"do_sample":false,"max_tokens":2048,'
            '"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
        ),
        "--stream",
        "--eval-batch-size",
        "1",
        "--timeout",
        "7200",
        "--limit",
        "8",
        "--datasets",
        "humaneval",
        "--dataset-args",
        '{"humaneval":{}}',
        "--work-dir",
        str(tmp_path),
        "--no-timestamp",
    ]


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.external_service("evalscope")
def test_qwen38_flash_next_int8_pp2_tp4_humaneval_evalscope_server() -> None:
    config = load_config(DEFAULT_CONFIG, CONFIG_ENV)
    run_evalscope_server_test(
        config,
        model_env=MODEL_ENV,
        model_label="Qwen3.8 Flash Next INT8 PP2+TP4+AITER+MTP3",
        required_hcu_count=8,
    )
