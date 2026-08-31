# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4-Flash-0731 Channel-FP8/INT8 DSpark runtime gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import (
    require_model_runtime,
    run_vllm_case,
)


MODEL_PATH = "DeepSeek-V4-Flash-0731-Channel-FP8-w8a8"
MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL"
INT8_MODEL_PATH = "DeepSeek-V4-Flash-0731-Channel-INT8-w8a8"
INT8_MODEL_ENV = "VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL"

pytestmark = [
    pytest.mark.model,
    pytest.mark.multi_hcu,
    pytest.mark.slow,
]


def _require_checkpoint(
    resources: HcuTestResources,
    *,
    hcu_count: int,
) -> Path:
    return require_model_runtime(
        resources,
        env_name=MODEL_ENV,
        relative_path=MODEL_PATH,
        label="DeepSeek-V4-Flash-0731 Channel-FP8 DSpark",
        hcu_count=hcu_count,
    )


def _require_int8_checkpoint(
    resources: HcuTestResources,
    *,
    hcu_count: int,
) -> Path:
    return require_model_runtime(
        resources,
        env_name=INT8_MODEL_ENV,
        relative_path=INT8_MODEL_PATH,
        label="DeepSeek-V4-Flash-0731 Channel-INT8 DSpark",
        hcu_count=hcu_count,
    )


def _assert_runtime_result(
    result: dict,
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
) -> None:
    assert result["speculative_method"] == "dspark"
    assert result["draft_token_count"] == 7
    assert result["pcp_world_size"] == 1
    assert result["requested_tensor_parallel_size"] == tensor_parallel_size
    assert result["requested_data_parallel_size"] == data_parallel_size
    assert result["output"]
    assert all(item["token_ids"] for item in result["output"])


def _assert_unified_deepep_execution(result: dict) -> None:
    assert result["requested_all2all_backend"] == "deepep_auto"
    log_text = Path(result["_log_path"]).read_text(
        encoding="utf-8",
        errors="replace",
    )
    assert "Using DeepEP auto MoE kernel with HT/LL experts." in log_text
    assert (
        "DeepEP auto selected contiguous high-throughput experts for this forward."
        in log_text
    )
    assert (
        "DeepEP auto selected masked low-latency experts for this forward."
        in log_text
    )


@pytest.mark.hcu_count(8)
def test_deepseek_v4_flash_dspark_tp8(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = _require_checkpoint(hcu_test_resources, hcu_count=8)
    result = run_vllm_case(
        "deepseek-v4-dspark-smoke",
        model_path,
        timeout_s=7200,
        extra_args=[
            "--topology",
            "tp8",
            "--gpu-memory-utilization",
            "0.9",
        ],
        log_label="deepseek-v4-dspark-tp8",
    )
    _assert_runtime_result(
        result,
        tensor_parallel_size=8,
        data_parallel_size=1,
    )


@pytest.mark.hcu_count(8)
def test_deepseek_v4_flash_dspark_dp8_ep8(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = _require_checkpoint(hcu_test_resources, hcu_count=8)
    result = run_vllm_case(
        "deepseek-v4-dspark-smoke",
        model_path,
        timeout_s=7200,
        extra_args=[
            "--topology",
            "dp8_ep8",
            "--gpu-memory-utilization",
            "0.9",
        ],
        log_label="deepseek-v4-dspark-dp8-ep8",
    )
    _assert_runtime_result(
        result,
        tensor_parallel_size=1,
        data_parallel_size=8,
    )
    _assert_unified_deepep_execution(result)


@pytest.mark.hcu_count(8)
def test_deepseek_v4_flash_int8_dspark_tp8(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = _require_int8_checkpoint(hcu_test_resources, hcu_count=8)
    result = run_vllm_case(
        "deepseek-v4-dspark-smoke",
        model_path,
        timeout_s=7200,
        extra_args=[
            "--topology",
            "tp8",
            "--gpu-memory-utilization",
            "0.9",
        ],
        log_label="deepseek-v4-int8-dspark-tp8",
    )
    _assert_runtime_result(
        result,
        tensor_parallel_size=8,
        data_parallel_size=1,
    )


@pytest.mark.hcu_count(8)
def test_deepseek_v4_flash_int8_dspark_dp8_ep8(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = _require_int8_checkpoint(hcu_test_resources, hcu_count=8)
    result = run_vllm_case(
        "deepseek-v4-dspark-smoke",
        model_path,
        timeout_s=7200,
        extra_args=[
            "--topology",
            "dp8_ep8",
            "--gpu-memory-utilization",
            "0.9",
        ],
        log_label="deepseek-v4-int8-dspark-dp8-ep8",
    )
    _assert_runtime_result(
        result,
        tensor_parallel_size=1,
        data_parallel_size=8,
    )
    _assert_unified_deepep_execution(result)
