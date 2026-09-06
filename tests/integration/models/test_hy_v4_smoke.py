# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eight-HCU ModelOpt MXFP8 and native-MTP coverage for HY V4."""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import require_model_runtime, run_vllm_case


HY_V4_MODEL = "/models/Hy4-preview-FP8-Testing"

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.model,
    pytest.mark.multi_hcu,
    pytest.mark.hcu_count(8),
    pytest.mark.slow,
]


def _assert_completion(record: dict[str, Any]) -> None:
    assert record["prompt_token_count"] > 0
    assert 1 <= len(record["token_ids"]) <= 8
    assert isinstance(record["text"], str)
    assert record["finish_reason"] in {"length", "stop", "eos"}
    cumulative_logprob = record["cumulative_logprob"]
    assert cumulative_logprob is None or math.isfinite(cumulative_logprob)


def test_hy_v4_blockwise_pure_tp8_triton_greedy_generation(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_HY_V4_MODEL",
        relative_path=HY_V4_MODEL,
        label="HY V4 ModelOpt MXFP8 pure TP8",
        hcu_count=8,
    )

    result = run_vllm_case(
        "tp-ep-smoke",
        model_path,
        timeout_s=7200,
        log_label="hy-v4-blockwise-tp8-triton",
        extra_env={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        },
        extra_args=[
            "--tensor-parallel-size",
            "8",
            "--disable-expert-parallel",
            "--gpu-memory-utilization",
            "0.95",
            "--moe-backend",
            "triton",
        ],
    )

    assert result["requested_tensor_parallel_size"] == 8
    assert result["requested_enable_expert_parallel"] is False
    assert result["requested_moe_backend"] == "triton"
    assert result["parallel_config"]["tensor_parallel_size"] == 8
    assert result["parallel_config"]["enable_expert_parallel"] is False
    assert len(result["output"]) == 2
    for record in result["output"]:
        _assert_completion(record)


def test_hy_v4_blockwise_pure_tp8_mtp_three_token_parity(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_HY_V4_MODEL",
        relative_path=HY_V4_MODEL,
        label="HY V4 ModelOpt MXFP8 pure TP8 native MTP",
        hcu_count=8,
    )

    result = run_vllm_case(
        "mtp-parity",
        model_path,
        timeout_s=7200,
        gpu_memory_utilization=0.95,
        log_label="hy-v4-blockwise-tp8-mtp3",
        extra_env={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        },
        extra_args=[
            "--tensor-parallel-size",
            "8",
            "--disable-expert-parallel",
            "--moe-backend",
            "triton",
            "--num-speculative-tokens",
            "3",
        ],
    )
    baseline_tokens = [record["token_ids"] for record in result["baseline"]]
    speculative_tokens = [
        record["token_ids"] for record in result["speculative"]
    ]
    assert speculative_tokens == baseline_tokens
    assert all(speculative_tokens)
    for record in [*result["baseline"], *result["speculative"]]:
        _assert_completion(record)


def test_hy_v4_fp8_e4m3_kv_cache_prefill_decode_and_mtp_parity(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_HY_V4_MODEL",
        relative_path=HY_V4_MODEL,
        label="HY V4 FP8 E4M3 KV-cache parity",
        hcu_count=8,
    )

    result = run_vllm_case(
        "hy-v4-kv-cache-parity",
        model_path,
        timeout_s=7200,
        gpu_memory_utilization=0.95,
        log_label="hy-v4-fp8-e4m3-kv-cache-parity",
        extra_env={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        },
        extra_args=[
            "--tensor-parallel-size",
            "8",
            "--disable-expert-parallel",
            "--moe-backend",
            "triton",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--num-speculative-tokens",
            "3",
        ],
    )

    baseline_tokens = [record["token_ids"] for record in result["baseline"]]
    quantized_tokens = [record["token_ids"] for record in result["quantized"]]
    quantized_mtp_tokens = [
        record["token_ids"] for record in result["quantized_mtp"]
    ]
    assert result["kv_cache_dtype"] == "fp8_e4m3"
    assert quantized_tokens == baseline_tokens
    assert quantized_mtp_tokens == baseline_tokens
    assert all(baseline_tokens)
    for record in [
        *result["baseline"],
        *result["quantized"],
        *result["quantized_mtp"],
    ]:
        _assert_completion(record)
