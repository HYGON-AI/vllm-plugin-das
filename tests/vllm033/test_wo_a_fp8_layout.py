"""Native WO-A output bytes and the actual Channel-FP8 WO-B consumer."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import wo_a_layout as layout
from hcu_vllm_das.adapter_api import adapter_module


def test_channel_fp8_admission_and_fallback(monkeypatch):
    from hcu_vllm_das.quantization.ct_w4a8 import HcuDSV41CTW4A8Config

    fp8 = adapter_module('dsv41_channel_fp8').HcuDSV41ChannelFP8Config
    monkeypatch.setattr(layout, '_BLAS_OVERRIDDEN', False)
    monkeypatch.setattr(layout, 'qualify_wo_a_layout', lambda _: True)
    cfg = NS(quant_config=fp8.__new__(fp8),
             model_config=NS(hf_config=NS(model_type='deepseek_v41'),
                             hf_text_config=NS(model_type='deepseek_v41', hidden_size=5120)),
             parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1,
                                prefill_context_parallel_size=1, data_parallel_size=8,
                                enable_expert_parallel=True))
    assert layout.configured(cfg, 'cpu')
    cfg.quant_config = HcuDSV41CTW4A8Config.__new__(HcuDSV41CTW4A8Config)
    assert layout.configured(cfg, 'cpu')
    cfg.quant_config = object()
    assert not layout.configured(cfg, 'cpu')
    cfg.quant_config = fp8.__new__(fp8)
    cfg.parallel_config.tensor_parallel_size = 2
    assert not layout.configured(cfg, 'cpu')
    cfg.parallel_config.tensor_parallel_size = 1
    monkeypatch.setattr(layout, '_BLAS_OVERRIDDEN', True)
    assert not layout.configured(cfg, 'cpu')
