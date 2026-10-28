# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real HCU BF16-head versus ConvRot INT8 operator accuracy."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytestmark = pytest.mark.hcu


@pytest.fixture(scope="module")
def hcu_head():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("a live HCU/HIP device is required")
    if not hasattr(torch.cuda.get_device_properties(0), "gcnArchName"):
        pytest.skip("the active device is not an HCU/HIP device")
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm_hcu.model_executor.layers.quantization.lm_head_convrot_int8 import ConvRotINT8LMHeadMethod

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
        patch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "1")
        with torch.inference_mode():
            torch.manual_seed(20261027)
            layer = nn.Module()
            layer.weight = nn.Parameter(
                torch.randn(4096, 5120, device="cuda", dtype=torch.bfloat16) * 0.05,
                requires_grad=False,
            )
            original_bf16_weight = layer.weight.detach().clone()
            method = ConvRotINT8LMHeadMethod(UnquantizedEmbeddingMethod())
            method.process_weights_after_loading(layer)
            assert original_bf16_weight.dtype == torch.bfloat16
            assert layer.weight.dtype == torch.int8
        yield layer, method, original_bf16_weight


@pytest.mark.parametrize("m", [1, 4, 16, 24])
def test_hcu_convrot_logits_against_original_bf16_head(hcu_head, m):
    layer, method, original_bf16_weight = hcu_head
    torch.manual_seed(20261027 + m)
    hidden = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16) * 0.05
    with torch.inference_mode():
        reference = F.linear(hidden, original_bf16_weight)
        eager = method.apply(layer, hidden)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, hidden)
        graph.replay()
        torch.cuda.synchronize()

        for mode, actual in (("eager", eager), ("graph", captured)):
            assert actual.shape == reference.shape == (m, 4096)
            assert actual.dtype == reference.dtype == torch.bfloat16
            assert torch.isfinite(reference).all().item()
            assert torch.isfinite(actual).all().item()
            reference_fp32 = reference.float()
            diff = actual.float() - reference_fp32
            metrics = {
                "max_abs_diff": diff.abs().max().item(),
                "mean_abs_diff": diff.abs().mean().item(),
                "relative_l2": (diff.norm() / reference_fp32.norm().clamp_min(1e-12)).item(),
                "top1_agreement": (actual.argmax(-1) == reference.argmax(-1)).float().mean().item(),
            }
            print(f"M={m} mode={mode} BF16 vs ConvRot INT8: {metrics}")
            # This is the same synthetic-input regression bound as the CPU
            # reference test, not a model-task or deployment accuracy gate.
            assert metrics["relative_l2"] < 0.03, metrics


def test_hcu_zero_hidden_states_match_original_bf16_head(hcu_head):
    layer, method, original_bf16_weight = hcu_head
    hidden = torch.zeros(4, 5120, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = F.linear(hidden, original_bf16_weight)
        actual = method.apply(layer, hidden)
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
