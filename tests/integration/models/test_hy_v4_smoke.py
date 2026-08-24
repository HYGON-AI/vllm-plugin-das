# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eight-HCU accuracy-first smoke coverage for the HY V4 checkpoint."""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import require_model_runtime, run_vllm_case


HY_V4_MODEL = "/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2"

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.model,
    pytest.mark.multi_hcu,
    pytest.mark.hcu_count(8),
    pytest.mark.slow,
]


def _assert_completion(record: dict[str, Any]) -> None:
    assert record["prompt_token_count"] > 0
    assert 1 <= len(record["token_ids"]) <= 4
    assert isinstance(record["text"], str)
    assert record["finish_reason"] in {"length", "stop", "eos"}
    cumulative_logprob = record["cumulative_logprob"]
    assert cumulative_logprob is None or math.isfinite(cumulative_logprob)


def test_hy_v4_tp8_ep8_triton_greedy_generation(
    hcu_test_resources: HcuTestResources,
) -> None:
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_HY_V4_MODEL",
        relative_path=HY_V4_MODEL,
        label="HY V4 FP8 W8A8 TP8+EP8",
        hcu_count=8,
    )

    result = run_vllm_case(
        "tp-ep-smoke",
        model_path,
        timeout_s=7200,
        log_label="hy-v4-tp8-ep8-triton",
        extra_args=[
            "--tensor-parallel-size",
            "8",
            "--gpu-memory-utilization",
            "0.95",
            "--moe-backend",
            "triton",
        ],
    )

    assert result["requested_tensor_parallel_size"] == 8
    assert result["requested_enable_expert_parallel"] is True
    assert result["requested_moe_backend"] == "triton"
    assert result["parallel_config"]["tensor_parallel_size"] == 8
    assert result["parallel_config"]["enable_expert_parallel"] is True
    assert len(result["output"]) == 2
    for record in result["output"]:
        _assert_completion(record)
