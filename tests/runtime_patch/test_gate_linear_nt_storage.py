# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import pytest
import torch

from vllm_hcu.utils.gate_linear_layout import (
    HCU_GATE_LINEAR_MARKER,
    prepare_gate_linear_nt_weight,
)


class Gate(torch.nn.Module):
    def __init__(self, shape=(6, 4), dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype),
            requires_grad=False,
        )
        self.allow_cublas_router_gemm = True
        self.out_dtype = torch.float32
        setattr(self, HCU_GATE_LINEAR_MARKER, True)


def test_prepare_preserves_values_parameter_and_loader_attributes():
    gate = Gate()
    weight = gate.weight
    weight.input_dim = 1
    weight.output_dim = 0
    loader = object()
    weight.weight_loader = loader
    values = weight.clone()
    assert prepare_gate_linear_nt_weight(gate, use_nn=True)
    assert gate.weight is weight
    assert weight.shape == (6, 4) and weight.stride() == (1, 6)
    assert weight.dtype == torch.bfloat16
    assert weight.input_dim == 1 and weight.output_dim == 0
    assert weight.weight_loader is loader
    assert torch.equal(weight, values)
    assert torch.equal(gate.state_dict()['weight'], values)
    x = torch.arange(18).reshape(3, 6).float()
    assert torch.equal(x @ weight.float(), x @ values.float())


def test_second_post_load_does_not_allocate_another_storage():
    gate = Gate()
    assert prepare_gate_linear_nt_weight(gate, use_nn=True)
    pointer = gate.weight.data_ptr()
    assert not prepare_gate_linear_nt_weight(gate, use_nn=True)
    assert gate.weight.data_ptr() == pointer


@pytest.mark.parametrize('raw_data', [False, True])
def test_in_place_reload_updates_the_parameter_used_by_forward(raw_data):
    gate = Gate()
    prepare_gate_linear_nt_weight(gate, use_nn=True)
    pointer = gate.weight.data_ptr()
    checkpoint = torch.arange(24, 48).reshape(4, 6).to(torch.bfloat16)
    with torch.no_grad():
        target = gate.weight.data if raw_data else gate.weight
        target.copy_(checkpoint.t())
    assert gate.weight.data_ptr() == pointer
    assert gate.weight.stride() == (1, 6)
    assert torch.equal(gate.weight, checkpoint.t())
    assert not prepare_gate_linear_nt_weight(gate, use_nn=True)


def test_replaced_weight_can_be_prepared_again():
    gate = Gate()
    prepare_gate_linear_nt_weight(gate, use_nn=True)
    replacement = torch.nn.Parameter(torch.ones(6, 4, dtype=torch.bfloat16), requires_grad=False)
    gate.weight = replacement
    assert prepare_gate_linear_nt_weight(gate, use_nn=True)
    assert gate.weight is replacement and gate.weight.stride() == (1, 6)
    assert torch.equal(gate.weight, torch.ones_like(gate.weight))


@pytest.mark.parametrize('reason', ['nn_disabled', 'ordinary_linear', 'not_eligible', 'bf16_output', 'fp32_weight', 'meta'])
def test_other_paths_are_untouched(reason):
    gate = Gate()
    use_nn = True
    if reason == 'nn_disabled':
        use_nn = False
    elif reason == 'ordinary_linear':
        delattr(gate, HCU_GATE_LINEAR_MARKER)
    elif reason == 'not_eligible':
        gate.allow_cublas_router_gemm = False
    elif reason == 'bf16_output':
        gate.out_dtype = torch.bfloat16
    elif reason == 'fp32_weight':
        gate.weight = torch.nn.Parameter(gate.weight.float(), requires_grad=False)
    else:
        gate.weight = torch.nn.Parameter(torch.empty(6, 4, device='meta', dtype=torch.bfloat16), requires_grad=False)
    weight, pointer = gate.weight, gate.weight.data_ptr()
    stride = weight.stride()
    assert not prepare_gate_linear_nt_weight(gate, use_nn=use_nn)
    assert gate.weight is weight and gate.weight.data_ptr() == pointer
    assert gate.weight.stride() == stride


def test_unquantized_post_load_prepares_gate_on_accelerator_path(monkeypatch):
    from vllm_hcu.model_executor.layers import linear
    from vllm_hcu.platforms import envs

    monkeypatch.setattr(linear.current_platform, 'is_cpu', lambda: False)
    monkeypatch.setattr(envs, 'VLLM_USE_NN', True)
    gate = Gate()
    linear.UnquantizedLinearMethod().process_weights_after_loading(gate)
    assert gate.weight.stride() == (1, 6)


def test_unquantized_post_load_preserves_cpu_dispatch(monkeypatch):
    from vllm_hcu.model_executor.layers import linear
    from vllm.model_executor.layers import utils

    monkeypatch.setattr(linear.current_platform, 'is_cpu', lambda: True)
    calls = []
    monkeypatch.setattr(utils, 'dispatch_cpu_unquantized_gemm', lambda layer, **kw: calls.append((layer, kw)))
    gate = Gate()
    linear.UnquantizedLinearMethod().process_weights_after_loading(gate)
    assert calls == [(gate, {'remove_weight': True})]
    assert gate.weight.stride() == (4, 1)
