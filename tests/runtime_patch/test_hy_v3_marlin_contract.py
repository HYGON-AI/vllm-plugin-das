# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin import (
    get_w8a8_int8_marlin_weights,
)


SOURCE = Path(
    "vllm_hcu/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe_marlin.py"
)


def test_marlin_packs_all_experts_like_individual_experts() -> None:
    weights = torch.arange(3 * 5 * 128, dtype=torch.int16).to(torch.int8)
    weights = weights.reshape(3, 5, 128)

    packed = get_w8a8_int8_marlin_weights(weights)
    reference = torch.stack(
        [get_w8a8_int8_marlin_weights(weight) for weight in weights]
    )

    assert packed.shape == (3, 2, 5 * 64)
    assert torch.equal(packed, reference)


def test_marlin_rejects_unaligned_source_k() -> None:
    with pytest.raises(ValueError, match=r"K \(65\).+k_tile \(64\)"):
        get_w8a8_int8_marlin_weights(torch.empty(2, 3, 65, dtype=torch.int8))


def test_fp8_marlin_returns_distinct_output_for_shared_experts() -> None:
    """HY3 aliases x as shared_experts_input, so Marlin must not mutate x."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CompressedTensorsW8A8FP8MarlinMoEMethod"
    )
    forward = next(
        node
        for node in method.body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_moe_forward"
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fused_experts_impl_fp8_marlin"
    ]

    assert len(calls) == 1
    inplace = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "inplace"
    )
    assert isinstance(inplace, ast.Constant)
    assert inplace.value is False


def test_fp8_marlin_weight_packing_does_not_requantize_checkpoint() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "fp32_to_fp8_e4m3fn" not in source
