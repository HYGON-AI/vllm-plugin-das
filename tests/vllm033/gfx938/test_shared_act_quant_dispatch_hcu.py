# SPDX-License-Identifier: Apache-2.0
"""Check actual CustomOp selection rather than class-level CUDA methods."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.moe import shared_act_quant as fusion


pytestmark = pytest.mark.hcu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")
@pytest.mark.parametrize("custom_ops", [
    ["all", "-quant_fp8"], ["all", "-silu_and_mul_with_clamp"],
])
def test_wrapper_does_not_bypass_disabled_custom_op(shared_fp8_selected_ops, monkeypatch, custom_ops):
    quant, act = shared_fp8_selected_ops(custom_ops)
    down = torch.nn.Module()
    down.scheme = SimpleNamespace(fp8_linear=SimpleNamespace(quant_fp8=quant))
    monkeypatch.setattr(fusion, "get_input_quant_key", lambda _: fusion.kFp8DynamicTokenSym)
    wrapper = fusion.QuantizedSharedActivation(act, down)
    wrapper._quantize = lambda *a, **kw: pytest.fail("disabled CustomOp was fused")
    x = torch.empty((7, 256), device="cuda", dtype=torch.bfloat16)
    x[:, :128] = 0.0625
    x[:, 128:] = 0.0009765625
    actual = wrapper(x)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual.view(torch.uint8), act(x).view(torch.uint8),
                               rtol=0, atol=0)
    actual_q, actual_s = quant(actual)
    expected_q, expected_s = quant(act(x))
    torch.testing.assert_close(actual_q.view(torch.uint8), expected_q.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(actual_s.view(torch.uint8), expected_s.view(torch.uint8),
                               rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")
def test_wrapper_rechecks_dispatch_after_installation(shared_fp8_selected_ops, monkeypatch):
    quant, act = shared_fp8_selected_ops(["all"])
    down = torch.nn.Module()
    down.scheme = SimpleNamespace(fp8_linear=SimpleNamespace(quant_fp8=quant))
    monkeypatch.setattr(fusion, "get_input_quant_key", lambda _: fusion.kFp8DynamicTokenSym)
    wrapper = fusion.QuantizedSharedActivation(act, down)
    wrapper._quantize = lambda *a, **kw: pytest.fail("changed CustomOp was fused")
    quant._forward_method = quant.forward_native
    x = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    assert isinstance(wrapper(x), torch.Tensor)
