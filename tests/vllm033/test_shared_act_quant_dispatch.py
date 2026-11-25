# SPDX-License-Identifier: Apache-2.0
"""Check actual CustomOp selection rather than class-level CUDA methods."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.moe import shared_act_quant as fusion


@pytest.mark.parametrize("custom_ops,quant_ok,act_ok", [
    (["all"], True, True),
    (["all", "-quant_fp8"], False, True),
    (["all", "-silu_and_mul_with_clamp"], True, False),
    (["all", "-quant_fp8", "-silu_and_mul_with_clamp"], False, False),
])
def test_actual_custom_op_dispatch(shared_fp8_selected_ops, custom_ops, quant_ok, act_ok):
    quant, act = shared_fp8_selected_ops(custom_ops)
    contract = fusion._quant_contract(SimpleNamespace(quant_fp8=quant))
    assert (contract is not None) is quant_ok
    if quant_ok:
        assert contract[0] == "lightop"
    else:
        assert quant._forward_method.__func__ is type(quant).forward_native
    assert fusion._activation_contract(act) is act_ok


@pytest.mark.parametrize("kind", ["native", "wrapped", "instance_cuda"])
def test_unknown_or_replaced_dispatch_keeps_original(shared_fp8_selected_ops, monkeypatch, kind):
    quant, act = shared_fp8_selected_ops(["all"])
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
