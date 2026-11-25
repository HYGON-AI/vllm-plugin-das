# SPDX-License-Identifier: Apache-2.0
"""Owned Channel-FP8 Q-norm consumer using native QuantizedActivation.

Core retains attention control flow and both original projection consumers.
Only the BF16 normalization/quantization producer is fused and shared.
"""
import os

import torch
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel

from .adapter_api import optional_adapter_module
from .moe.shared_act_quant import (
    TokenFP8TritonLinearKernel,_quant_contract,enable_shared_prequant,
)


def _kernel(layer):
    return getattr(getattr(layer,'scheme',None),'fp8_linear',None)


def configure(attention,config):
    """Bind only compatible native consumers; INT8/MXFP8/PCP stay native."""
    if os.getenv('VLLM_HCU_USE_CUSTOM_OPS','1').lower() in ('0','false','off'):
        return None
    parallel=config.parallel_config
    if (parallel.tensor_parallel_size!=1
            or parallel.prefill_context_parallel_size!=1
            or parallel.decode_context_parallel_size!=1
            or attention.use_pcp or attention.kv_mxfp8
            or attention._wq_b_uses_aiter_block_scaled):
        return None
    from .fp4_indexer import Fp4Attention,Fp4Indexer
    if type(attention) is not Fp4Attention:
        return None
    indexer=attention.indexer
    if indexer is not None and type(indexer) is not Fp4Indexer:
        return None
    layers=[attention.wq_b]+([indexer.wq_b] if indexer is not None else [])
    contracts=[]
    for layer in layers:
        kernel=_kernel(layer)
        if (type(kernel) not in (TritonFP8ScaledMMLinearKernel,TokenFP8TritonLinearKernel)
                or kernel.config.activation_quant_key!=kFp8DynamicTokenSym
                or kernel.fp8_dtype!=torch.float8_e4m3fn
                or getattr(layer,'requires_unquantized_input',False)):
            return None
        contract=_quant_contract(kernel)
        if contract is None:return None
        contracts.append(contract)
    if len(set(contracts))!=1:return None
    module=optional_adapter_module('hcu_adapter.kernels.qkv_norm_token_fp8_triton')
    if module is None:return None
    if not all(enable_shared_prequant(layer) for layer in layers):return None
    return tuple(layers),contracts[0],module


def try_split(attention,x):
    binding=getattr(attention,'_hcu_qnorm_token_fp8',None)
    if binding is None:return None
    layers,contract,module=binding
    if (any(type(_kernel(layer)) is not TokenFP8TritonLinearKernel
            or get_input_quant_key(layer)!=kFp8DynamicTokenSym
            or _quant_contract(_kernel(layer))!=contract for layer in layers)
            or x.ndim!=2 or x.shape[1]!=attention.q_lora_rank+attention.head_dim
            or (contract[0]=='lightop' and attention.q_lora_rank%16)):
        return None
    qw,kw=attention.q_norm.weight.data,attention.kv_norm.weight.data
    if not module.supported(x,qw,kw,attention.eps):return None
    lightop=contract[0]=='lightop' and not torch.compiler.is_compiling()
    if lightop and not contract[1] and torch.cuda.is_current_stream_capturing():
        lightop=False
    q,scale,kv=module.norm_quantize(x,qw,kw,attention.eps,lightop=lightop)
    return QuantizedActivation(q,scale,x.dtype,
        torch.Size((x.shape[0],attention.q_lora_rank)),kFp8DynamicTokenSym),None,kv
