# SPDX-License-Identifier: Apache-2.0
"""Capability publication and native fallback with real consumer classes."""
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def consumers():
    from vllm.config import CompilationConfig,VllmConfig,set_current_vllm_config
    from vllm.plugins import load_general_plugins
    from hcu_vllm_das.fp4_indexer import Fp4Attention,Fp4Indexer
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym,kFp8StaticChannelSym
    load_general_plugins()
    cfg=VllmConfig(compilation_config=CompilationConfig(mode=0,custom_ops=['all']))
    with set_current_vllm_config(cfg):
        def layer():
            c=FP8ScaledMMLinearLayerConfig(
                weight_quant_key=kFp8StaticChannelSym,activation_quant_key=kFp8DynamicTokenSym,
                weight_shape=(64,1280),input_dtype=torch.bfloat16,out_dtype=torch.bfloat16)
            obj=torch.nn.Module()
            obj.scheme=SimpleNamespace(fp8_linear=TritonFP8ScaledMMLinearKernel(
                c,('weight','weight_scale','input_scale','input_scale_ub')))
            return obj
        attention=Fp4Attention.__new__(Fp4Attention);torch.nn.Module.__init__(attention)
        attention.use_pcp=attention.kv_mxfp8=attention._wq_b_uses_aiter_block_scaled=False
        attention.q_lora_rank=1280;attention.head_dim=512;attention.eps=1e-20
        attention.wq_b=layer()
        indexer=Fp4Indexer.__new__(Fp4Indexer);torch.nn.Module.__init__(indexer)
        indexer.wq_b=layer();attention.indexer=indexer
        parallel=SimpleNamespace(tensor_parallel_size=1,prefill_context_parallel_size=1,
                                 decode_context_parallel_size=1)
        yield attention,SimpleNamespace(parallel_config=parallel)


@pytest.mark.parametrize('has_indexer',[False,True])
def test_actual_native_consumer_capability_and_cpu_fallback(consumers,has_indexer):
    from hcu_vllm_das.qnorm_token_fp8 import configure,try_split
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    a,cfg=consumers
    if not has_indexer:a.indexer=None
    a._hcu_qnorm_token_fp8=configure(a,cfg)
    assert a._hcu_qnorm_token_fp8 is not None
    assert get_input_quant_key(a.wq_b)==kFp8DynamicTokenSym
    if has_indexer:assert get_input_quant_key(a.indexer.wq_b)==kFp8DynamicTokenSym
    a.q_norm=torch.nn.Module();a.kv_norm=torch.nn.Module()
    a.q_norm.weight=torch.nn.Parameter(torch.ones(1280),requires_grad=False)
    a.kv_norm.weight=torch.nn.Parameter(torch.ones(512),requires_grad=False)
    with torch.inference_mode():assert try_split(a,torch.ones(7,1792,dtype=torch.bfloat16)) is None


@pytest.mark.parametrize('attribute',['use_pcp','kv_mxfp8','_wq_b_uses_aiter_block_scaled'])
def test_other_native_attention_contracts_are_not_rebound(consumers,attribute):
    from hcu_vllm_das.qnorm_token_fp8 import configure
    a,cfg=consumers;before=a.wq_b.scheme.fp8_linear
    setattr(a,attribute,True)
    assert configure(a,cfg) is None
    assert a.wq_b.scheme.fp8_linear is before


@pytest.mark.parametrize('has_indexer', [False, True])
def test_backend_owned_normalization_keeps_both_native_consumers(consumers, monkeypatch, has_indexer):
    from hcu_vllm_das.qnorm_token_fp8 import configure
    a, cfg = consumers
    if not has_indexer:
        a.indexer = None
    monkeypatch.setattr(type(a), 'accepts_unnormed_unroped_query',
                        property(lambda self: True))
    layers = [a.wq_b] + ([a.indexer.wq_b] if has_indexer else [])
    before = [layer.scheme.fp8_linear for layer in layers]
    assert configure(a, cfg) is None
    assert all(layer.scheme.fp8_linear is previous
               for layer, previous in zip(layers, before))


@pytest.mark.parametrize('attribute',['tensor_parallel_size','prefill_context_parallel_size','decode_context_parallel_size'])
def test_other_parallel_layouts_keep_native(consumers,attribute):
    from hcu_vllm_das.qnorm_token_fp8 import configure
    a,cfg=consumers;setattr(cfg.parallel_config,attribute,2)
    assert configure(a,cfg) is None


def test_unknown_consumer_does_not_partially_mutate_main(consumers):
    from hcu_vllm_das.qnorm_token_fp8 import configure
    a,cfg=consumers;before=a.wq_b.scheme.fp8_linear
    a.indexer.wq_b.scheme.fp8_linear=object()
    assert configure(a,cfg) is None
    assert a.wq_b.scheme.fp8_linear is before


def test_changed_consumer_cannot_receive_quantized_values(consumers):
    from hcu_vllm_das.qnorm_token_fp8 import configure,try_split
    a,cfg=consumers;a._hcu_qnorm_token_fp8=configure(a,cfg)
    assert a._hcu_qnorm_token_fp8 is not None
    a.wq_b.scheme.fp8_linear=object()
    assert try_split(a,torch.ones(7,1792,dtype=torch.bfloat16)) is None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "NO", " 0 ", " false "])
def test_master_rollback_does_not_touch_qnorm_consumers(monkeypatch, value):
    from hcu_vllm_das.qnorm_token_fp8 import configure
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", value)
    assert configure(object(), object()) is None
