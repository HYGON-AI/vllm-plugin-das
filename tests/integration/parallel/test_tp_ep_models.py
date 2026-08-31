# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""TP + expert-parallel real-model smoke coverage."""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import (
    require_gfx_arch,
    require_model_runtime,
    run_vllm_case,
)


QWEN35_35B_A3B = "qwen3.5/Qwen3.5-35B-A3B"
DEEPSEEK_R1_CHANNEL_INT8 = "vllm-w8a8-models/DeepSeek-R1-0528-Channel-INT8"
GLM52_CHANNEL_INT8 = "GLM-5.2-Channel-INT8-w8a8"
GLM52_DEEPEP_MODES = (
    pytest.param("deepep_high_throughput", id="high-throughput"),
    pytest.param("deepep_low_latency", id="low-latency"),
)
QWEN35_35B_A3B_TP_EP_SIZES = (
    pytest.param(4, id="tp4-ep4"),
    pytest.param(2, id="tp2-ep2"),
)
QWEN35_35B_A3B_GPU_MEMORY_UTILIZATION = {
    2: 0.4,
    4: 0.4,
}
QWEN35_35B_A3B_TP_EP_MOE_PATHS = (
    pytest.param(
        "aiter-auto-shuffle",
        "aiter",
        {
            "VLLM_HCU_USE_AITER_MOE_SHUFFLE": "1",
        },
        id="aiter-auto-shuffle",
    ),
    pytest.param(
        "aiter-auto-nonshuffle",
        "aiter",
        {
            "VLLM_HCU_USE_AITER_MOE_SHUFFLE": "0",
        },
        id="aiter-auto-nonshuffle",
    ),
    pytest.param(
        "triton",
        "triton",
        {},
        id="triton",
    ),
)
DEEPSEEK_R1_CHANNEL_INT8_TP_EP_MOE_PATHS = (
    pytest.param(
        "auto",
        "auto",
        {},
        id="auto",
    ),
    pytest.param(
        "target-triton",
        "triton",
        {
            "VLLM_ROCM_USE_AITER_MOE": "0",
            "VLLM_HCU_USE_AITER_W8A8_FP8_MOE": "0",
        },
        id="target-triton",
    ),
    pytest.param(
        "deep-gemm",
        "deep_gemm",
        {},
        id="deep-gemm",
    ),
)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AssertionError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise AssertionError(f"{name} must be positive, got {parsed}")
    return parsed


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise AssertionError(f"{name} must be a float, got {value!r}") from exc
    if not 0 < parsed < 1:
        raise AssertionError(f"{name} must be in (0, 1), got {parsed}")
    return parsed


def _qwen35_gpu_memory_utilization(tp_size: int) -> float:
    default = QWEN35_35B_A3B_GPU_MEMORY_UTILIZATION[tp_size]
    common_name = "VLLM_HCU_QWEN35_35B_A3B_GPU_MEMORY_UTILIZATION"
    tp_name = f"VLLM_HCU_QWEN35_35B_A3B_TP{tp_size}_GPU_MEMORY_UTILIZATION"
    common = _float_env(common_name, default)
    return _float_env(tp_name, common)


def _assert_tp_ep_result(
    result: dict[str, Any],
    *,
    expected_tp: int,
    expected_dp: int,
    expected_gpu_memory_utilization: float,
    expected_all2all: str | None,
    expected_moe_backend: str,
) -> None:
    assert result["requested_tensor_parallel_size"] == expected_tp
    assert result["requested_data_parallel_size"] == expected_dp
    assert result["requested_all2all_backend"] == expected_all2all
    assert result["requested_enable_expert_parallel"] is True
    assert result["requested_gpu_memory_utilization"] == expected_gpu_memory_utilization
    assert result["requested_moe_backend"] == expected_moe_backend
    parallel_config = result["parallel_config"]
    if parallel_config:
        assert parallel_config["tensor_parallel_size"] == expected_tp
        assert parallel_config["data_parallel_size"] == expected_dp
        if expected_all2all is not None:
            assert parallel_config["all2all_backend"] == expected_all2all
        assert parallel_config["enable_expert_parallel"] is True
        assert parallel_config["world_size"] >= expected_tp
    assert len(result["output"]) == 2
    for record in result["output"]:
        assert record["prompt_token_count"] > 0
        assert 1 <= len(record["token_ids"]) <= 4
        assert record["finish_reason"] in {"length", "stop", "eos"}


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(4)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.parametrize("tp_size", QWEN35_35B_A3B_TP_EP_SIZES)
@pytest.mark.parametrize(
    ("path_label", "moe_backend", "moe_env"),
    QWEN35_35B_A3B_TP_EP_MOE_PATHS,
)
def test_qwen35_35b_a3b_tp_ep_smoke(
    hcu_test_resources: HcuTestResources,
    path_label: str,
    moe_backend: str,
    moe_env: dict[str, str],
    tp_size: int,
) -> None:
    require_gfx_arch("gfx938", "Qwen3.5-35B-A3B TP+EP")
    gpu_memory_utilization = _qwen35_gpu_memory_utilization(tp_size)
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_QWEN35_35B_A3B_MODEL",
        relative_path=QWEN35_35B_A3B,
        label="Qwen3.5-35B-A3B TP+EP",
        hcu_count=tp_size,
    )

    result = run_vllm_case(
        "tp-ep-smoke",
        model_path,
        timeout_s=3600,
        extra_env=moe_env,
        log_label=f"tp-ep-smoke-tp{tp_size}-ep{tp_size}-{path_label}",
        extra_args=[
            "--tensor-parallel-size",
            str(tp_size),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--moe-backend",
            moe_backend,
        ],
    )

    _assert_tp_ep_result(
        result,
        expected_tp=tp_size,
        expected_dp=1,
        expected_gpu_memory_utilization=gpu_memory_utilization,
        expected_all2all=None,
        expected_moe_backend=moe_backend,
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.parametrize(
    ("path_label", "moe_backend", "moe_env"),
    DEEPSEEK_R1_CHANNEL_INT8_TP_EP_MOE_PATHS,
)
def test_deepseek_r1_channel_int8_tp_ep_smoke(
    hcu_test_resources: HcuTestResources,
    path_label: str,
    moe_backend: str,
    moe_env: dict[str, str],
) -> None:
    require_gfx_arch("gfx938", "DeepSeek-R1-Channel-INT8 TP+EP")
    tp_size = _int_env("VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_TP", 8)
    gpu_memory_utilization = _float_env(
        "VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_GPU_MEMORY_UTILIZATION",
        0.6,
    )
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_MODEL",
        relative_path=DEEPSEEK_R1_CHANNEL_INT8,
        label="DeepSeek-R1-Channel-INT8 TP+EP",
        hcu_count=tp_size,
    )

    result = run_vllm_case(
        "tp-ep-smoke",
        model_path,
        timeout_s=5400,
        extra_env=moe_env,
        log_label=f"tp-ep-smoke-{path_label}",
        extra_args=[
            "--tensor-parallel-size",
            str(tp_size),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--moe-backend",
            moe_backend,
        ],
    )

    _assert_tp_ep_result(
        result,
        expected_tp=tp_size,
        expected_dp=1,
        expected_gpu_memory_utilization=gpu_memory_utilization,
        expected_all2all=None,
        expected_moe_backend=moe_backend,
    )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(8)
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.parametrize("all2all_backend", GLM52_DEEPEP_MODES)
def test_glm52_channel_int8_deepep_smoke(
    hcu_test_resources: HcuTestResources,
    all2all_backend: str,
) -> None:
    require_gfx_arch("gfx938", "GLM-5.2-Channel-INT8 DeepEP")
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_GLM52_CHANNEL_INT8_MODEL",
        relative_path=GLM52_CHANNEL_INT8,
        label="GLM-5.2-Channel-INT8 DeepEP",
        hcu_count=8,
    )

    result = run_vllm_case(
        "tp-ep-smoke",
        model_path,
        timeout_s=5400,
        log_label=f"glm52-int8-{all2all_backend}",
        extra_args=[
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "8",
            "--all2all-backend",
            all2all_backend,
            "--gpu-memory-utilization",
            "0.9",
            "--moe-backend",
            "deep_gemm",
        ],
    )

    _assert_tp_ep_result(
        result,
        expected_tp=1,
        expected_dp=8,
        expected_gpu_memory_utilization=0.9,
        expected_all2all=all2all_backend,
        expected_moe_backend="deep_gemm",
    )
