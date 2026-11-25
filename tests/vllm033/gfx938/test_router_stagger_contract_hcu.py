# SPDX-License-Identifier: Apache-2.0
"""Actual GateLinear ownership, live-weight reload and exact native fallback."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")]


@pytest.fixture
def consumer(monkeypatch):
    from vllm.config import CompilationConfig,VllmConfig,set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    monkeypatch.setattr(parallel_state, '_TP', SimpleNamespace(rank_in_group=0,world_size=1))
    config=VllmConfig(compilation_config=CompilationConfig(mode=0,custom_ops=['all']))
    with set_current_vllm_config(config):
        gate=GateLinear(5120,384,out_dtype=torch.float32,params_dtype=torch.bfloat16).cuda()
        cfg=SimpleNamespace(parallel_config=SimpleNamespace(
            tensor_parallel_size=1,prefill_context_parallel_size=1,
            decode_context_parallel_size=1))
        yield cfg,SimpleNamespace(gate=gate)


@pytest.mark.parametrize('return_bias',[False,True])
@torch.inference_mode()
def test_actual_gate_output_contract_and_instance_ownership(consumer,return_bias):
    from hcu_vllm_das.moe.router_stagger import bind_router
    cfg,moe=consumer;gate=moe.gate;gate.return_bias=return_bias
    original=type(gate).forward;state_keys=set(gate.state_dict())
    gate.weight.normal_(std=.014)
    # An unsupported contract must call the exact bound native method. The
    # injected module asserts that unsupported input cannot enter project().
    fake=SimpleNamespace(supported=lambda x,w:False,
                         project=lambda x,w:(_ for _ in ()).throw(AssertionError()))
    with patch('hcu_vllm_das.moe.router_stagger.optional_adapter_module',return_value=fake):
        assert bind_router(cfg,moe)
        assert bind_router(cfg,moe)
    x=torch.randn(7,5120,device='cuda',dtype=torch.bfloat16)
    ref=gate._hcu_stagger_router_binding[0](x);actual=gate(x)
    if return_bias:
        assert actual[1] is ref[1] is None
        actual,ref=actual[0],ref[0]
    torch.testing.assert_close(actual,ref,rtol=0,atol=0)
    assert type(gate).forward is original
    assert set(gate.state_dict())==state_keys


@pytest.mark.parametrize('dimension',['tensor_parallel_size','prefill_context_parallel_size','decode_context_parallel_size'])
def test_parallel_fallback(consumer,dimension):
    from hcu_vllm_das.moe.router_stagger import bind_router
    cfg,moe=consumer;original=moe.gate.forward
    setattr(cfg.parallel_config,dimension,2)
    assert not bind_router(cfg,moe)
    assert moe.gate.forward==original


def test_unknown_forward_retained(consumer):
    from hcu_vllm_das.moe.router_stagger import bind_router
    cfg,moe=consumer;moe.gate.forward=lambda x: x
    assert not bind_router(cfg,moe)


def test_existing_disable_switch(consumer,monkeypatch):
    from hcu_vllm_das.moe.router_stagger import bind_router
    cfg,moe=consumer;monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','0')
    assert not bind_router(cfg,moe)


@pytest.mark.parametrize('attribute,value',[('out_dtype',torch.bfloat16),
    ('allow_cublas_router_gemm',False),('is_unquantized',False),
    ('_vllm_hcu_deepseek_v4_gate',True)])
def test_other_native_semantics_stay_native(consumer,attribute,value):
    from hcu_vllm_das.moe.router_stagger import bind_router
    cfg,moe=consumer;setattr(moe.gate,attribute,value)
    assert not bind_router(cfg,moe)
