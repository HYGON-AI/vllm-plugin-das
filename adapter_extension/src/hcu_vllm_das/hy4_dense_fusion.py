# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional HY4 dense/shared FFN INT8 fusion; execution belongs to Adapter."""
import torch

from .adapter_api import adapter_module
from .environment import _flag, custom_ops_enabled
from .hy4_ihc_launch import _device_arch
from .quantization.hy4_ct_policy import is_hy4


def _supported_config(cfg):
    return (is_hy4(cfg.model_config.hf_config)
            and cfg.model_config.dtype == torch.bfloat16
            and cfg.lora_config is None and cfg.use_v2_model_runner
            and cfg.compilation_config.mode == 0 and _device_arch() == 'gfx938')


def enabled(cfg):
    master = custom_ops_enabled()
    child = _flag('HCU_VLLM_HY4_DENSE_FUSION', default=False)
    return master and child and _supported_config(cfg)


def attach(model, cfg):
    """After native construction, keep ignored/unquantized FFNs unchanged."""
    if not enabled(cfg):
        return model
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
    adapter = adapter_module('lightop_int8_linear')
    for layer in list(model.modules()):
        if not isinstance(layer, HYV4FeedForward) or type(layer.act_fn) is not SiluAndMul:
            continue
        down = layer.down_proj
        if (type(down.quant_method) is adapter.HcuLightopInt8LinearMethod
                and down.weight.dtype == torch.int8 and down.input_is_parallel):
            layer.act_fn = adapter.FusedSiluInt8Producer(layer.act_fn, down)
    return model
