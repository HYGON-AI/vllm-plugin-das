# SPDX-License-Identifier: Apache-2.0
"""SwiGLU prequantization for eligible V4.1 CT INT8 shared experts."""
from functools import partial
import os

import torch
from .adapter_api import adapter_module

HcuDSV41Int8LinearMethod = adapter_module('dsv41_int8_linear').HcuDSV41Int8LinearMethod
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP

_LOG = init_logger(__name__)
_ENV = 'HCU_DSV41_SHARED_EXPERT_INT8_FUSION'


def bind_shared_expert(config, moe):
    from .moe.shared_act_quant import bind_shared_expert_fp8

    if bind_shared_expert_fp8(config, moe):
        return
    if os.getenv(_ENV, '1') != '1':
        return
    _LOG.warning_once('HCU DSV4.1 shared expert INT8 fusion requested')
    parallel = config.parallel_config
    if (parallel.tensor_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not torch.cuda.is_available()):
        _LOG.warning_once('Shared expert INT8 fusion skipped for parallel or device configuration')
        return
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'gcnArchName', '')
    if arch.split(':', 1)[0] != 'gfx938':
        _LOG.warning_once('Shared expert INT8 fusion skipped for device architecture %s', arch)
        return
    mlp = moe.shared_experts
    if (not isinstance(mlp, DeepseekV4MLP) or moe.use_mega_moe
            or moe.fuse_heterogeneous_shared_expert
            or getattr(mlp.forward, '__func__', None) is not DeepseekV4MLP.forward):
        _LOG.warning_once('Shared expert INT8 fusion skipped for MLP %s (mega=%s, heterogeneous=%s)',
                       type(mlp).__name__, moe.use_mega_moe,
                       moe.fuse_heterogeneous_shared_expert)
        return
    down, act = mlp.down_proj, mlp.act_fn
    if (type(down) is not RowParallelLinear or down.tp_size != 1
            or down.bias is not None or not down.return_bias
            or down.input_size != 2304 or down.output_size != 5120
            or type(down.quant_method) is not HcuDSV41Int8LinearMethod
            or not isinstance(act, SiluAndMulWithClamp)
            or (act.swiglu_limit, act.alpha, act.beta) != (10.0, 1.0, 0.0)):
        _LOG.warning_once('Shared expert INT8 fusion skipped for linear %s and activation %s',
                       type(down.quant_method).__name__, type(act).__name__)
        return
    try:
        from lightop import fuse_silu_mul_clamp_quant
    except ImportError:
        _LOG.warning_once('Shared expert INT8 fusion skipped because LightOp clamp quantization is unavailable')
        return
    mlp.forward = partial(_forward, mlp, mlp.forward, fuse_silu_mul_clamp_quant)
    _LOG.warning_once('HCU DSV4.1 shared expert INT8 fusion bound to MLP')


def _forward(mlp, original, quantize, x):
    down = mlp.down_proj
    method = down.quant_method
    if (mlp._gateup_scale is not None or x.ndim != 2 or x.dtype != torch.bfloat16
            or not x.is_cuda or not 1 <= x.shape[0] <= 256
            or not method.supports_shared_expert_prequantized(down)):
        if x.ndim == 2 and 1 <= x.shape[0] <= 256:
            _LOG.warning_once('Shared expert INT8 fallback: shape=%s dtype=%s gateup=%s weight=%s scale=%s',
                           tuple(x.shape), x.dtype, mlp._gateup_scale is not None,
                           tuple(down.weight.shape), tuple(down.weight_scale.shape))
        return original(x)
    gate_up, _ = mlp.gate_up_proj(x)
    if (gate_up.ndim != 2 or gate_up.shape[1] != 4608
            or gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous()
            or gate_up.device != down.weight.device):
        output, _ = down(mlp.act_fn(gate_up))
        return output
    q, scale = quantize(gate_up, mlp.act_fn.swiglu_limit)
    output = method.apply_prequantized(down, q, scale)
    if torch.cuda.is_current_stream_capturing():
        _LOG.warning_once('HCU DSV4.1 shared expert INT8 fused SwiGLU quantization captured')
        return output
    _LOG.warning_once('HCU DSV4.1 shared expert INT8 fused SwiGLU quantization active')
    return output
