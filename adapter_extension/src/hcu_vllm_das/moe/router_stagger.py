# SPDX-License-Identifier: Apache-2.0
"""Instance-owned Channel-FP8 router projection; native selection is unchanged."""
from types import MethodType

import torch

from ..adapter_api import optional_adapter_module
from ..environment import fp8_fusions_enabled


def _eligible(gate):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    if type(gate) is not GateLinear:
        return False
    expected = getattr(GateLinear, '_vllm_hcu_original_forward', GateLinear.forward)
    current = GateLinear.forward
    known_forward = (current is expected or (
        getattr(current, '_vllm_hcu_nn_layout_wrapper', False)
        and getattr(current, '__wrapped__', None) is expected))
    return (known_forward and getattr(gate.forward, '__func__', None) is current
            and gate.is_unquantized and gate.allow_cublas_router_gemm
            and not gate.is_gfx950 and gate.bias is None
            and gate.out_dtype == torch.float32
            and not getattr(gate, '_vllm_hcu_deepseek_v4_gate', False)
            and not any(getattr(gate, name, False) for name in (
                'allow_ll_bf16_gemm', 'allow_fp32_router_gemm',
                'allow_ll_fp32w_gemm', 'allow_bf16x3_router_gemm',
                'allow_rocm_bf16x3_router_gemm')))


def _forward(gate, x):
    original, module = gate._hcu_stagger_router_binding
    # Re-evaluate mutable output / backend policy and tensor metadata. A new
    # weight allocation or an in-place checkpoint reload reads live weights.
    if (_eligible_bound(gate) and module.supported(x, gate.weight)):
        return gate._return(module.project(x, gate.weight))
    return original(x)


def _eligible_bound(gate):
    return (gate.is_unquantized and gate.allow_cublas_router_gemm
            and gate.bias is None and gate.out_dtype == torch.float32
            and not gate.is_gfx950
            and not getattr(gate, '_vllm_hcu_deepseek_v4_gate', False)
            and not any(getattr(gate, name, False) for name in (
                'allow_ll_bf16_gemm', 'allow_fp32_router_gemm',
                'allow_ll_fp32w_gemm', 'allow_bf16x3_router_gemm',
                'allow_rocm_bf16x3_router_gemm')))


def bind_router(config, moe):
    """Bind at the existing owned decoder seam without changing Core classes."""
    if not fp8_fusions_enabled():
        return False
    p = config.parallel_config
    if (p.tensor_parallel_size != 1 or p.prefill_context_parallel_size != 1
            or p.decode_context_parallel_size != 1):
        return False
    gate = getattr(moe, 'gate', None)
    if getattr(gate, '_hcu_stagger_router_binding', None) is not None:
        return getattr(gate.forward, '__func__', None) is _forward
    if not _eligible(gate):
        return False
    module = optional_adapter_module('hcu_adapter.kernels.router_stagger_triton')
    if module is None:
        return False
    gate._hcu_stagger_router_binding = (gate.forward, module)
    gate.forward = MethodType(_forward, gate)
    return True
