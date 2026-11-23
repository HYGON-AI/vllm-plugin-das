# SPDX-License-Identifier: Apache-2.0
"""Check actual CustomOp selection rather than class-level CUDA methods."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.moe import shared_act_quant as fusion


@pytest.fixture
def selected_ops(monkeypatch):
    from hcu_adapter.bridges.vllm_input_quant_lightop import _wrap
    from vllm.model_executor.layers.quantization import input_quant_fp8 as module

    original = module.QuantFP8.forward_cuda
    original = getattr(original, "__wrapped__", original)
    monkeypatch.setattr(module.QuantFP8, "forward_cuda", _wrap(module, original))

    def construct(custom_ops):
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

        with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(
                mode=0, custom_ops=custom_ops))):
            quant = module.QuantFP8(False, GroupShape.PER_TOKEN, compile_native=False)
            act = SiluAndMulWithClamp(10.0, compile_native=False)
        return quant, act

    return construct


@pytest.mark.parametrize("custom_ops,quant_ok,act_ok", [
    (["all"], True, True),
    (["all", "-quant_fp8"], False, True),
    (["all", "-silu_and_mul_with_clamp"], True, False),
    (["all", "-quant_fp8", "-silu_and_mul_with_clamp"], False, False),
])
def test_actual_custom_op_dispatch(selected_ops, custom_ops, quant_ok, act_ok):
    quant, act = selected_ops(custom_ops)
    contract = fusion._quant_contract(SimpleNamespace(quant_fp8=quant))
    assert (contract is not None) is quant_ok
    if quant_ok:
        assert contract[0] == "lightop"
    else:
        assert quant._forward_method.__func__ is type(quant).forward_native
    assert fusion._activation_contract(act) is act_ok


@pytest.mark.parametrize("kind", ["native", "wrapped", "instance_cuda"])
def test_unknown_or_replaced_dispatch_keeps_original(selected_ops, monkeypatch, kind):
    quant, act = selected_ops(["all"])
    for op in (quant, act):
        if kind == "native":
            monkeypatch.setattr(op, "_forward_method", op.forward_native)
        elif kind == "wrapped":
            original = op._forward_method
            monkeypatch.setattr(op, "_forward_method", lambda *args, _fn=original: _fn(*args))
        else:
            monkeypatch.setattr(op, "forward_cuda", lambda *args: None)
    assert fusion._quant_contract(SimpleNamespace(quant_fp8=quant)) is None
    assert not fusion._activation_contract(act)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")
@pytest.mark.parametrize("custom_ops", [
    ["all", "-quant_fp8"], ["all", "-silu_and_mul_with_clamp"],
])
def test_wrapper_does_not_bypass_disabled_custom_op(selected_ops, monkeypatch, custom_ops):
    quant, act = selected_ops(custom_ops)
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
def test_wrapper_rechecks_dispatch_after_installation(selected_ops, monkeypatch):
    quant, act = selected_ops(["all"])
    down = torch.nn.Module()
    down.scheme = SimpleNamespace(fp8_linear=SimpleNamespace(quant_fp8=quant))
    monkeypatch.setattr(fusion, "get_input_quant_key", lambda _: fusion.kFp8DynamicTokenSym)
    wrapper = fusion.QuantizedSharedActivation(act, down)
    wrapper._quantize = lambda *a, **kw: pytest.fail("changed CustomOp was fused")
    quant._forward_method = quant.forward_native
    x = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    assert isinstance(wrapper(x), torch.Tensor)
