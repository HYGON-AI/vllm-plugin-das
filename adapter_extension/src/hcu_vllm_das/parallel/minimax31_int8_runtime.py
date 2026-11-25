# SPDX-License-Identifier: Apache-2.0
"""Select an explicit correct backend for trace-workload INT8 geometries.

The installed LightOP channelwise wrapper always returns True, including when
hipBLASLt reports status 6 and leaves output unwritten. Qualified other output
widths retain that provider; N6144 uses Core's retained Triton implementation.
This only replaces per-consumer execution objects, never processed weights.
"""
import json
import torch


def select_qualified_int8(model, *, verify_transpose=False, verify_row_major=False):
    from hcu_vllm_das.quantization.minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
    from .minimax31_verify_int8 import VerifyTransposeInt8Kernel
    pending=[]
    for name,layer in model.named_modules():
        scheme=getattr(layer,'scheme',None);kernel=getattr(scheme,'kernel',None)
        if not isinstance(kernel,MiniMax31HipBlasLtInt8Kernel):continue
        w=layer.weight
        if w.ndim!=2 or w.dtype!=torch.int8 or w.stride()!=(1,w.shape[0]):
            raise ValueError('unexpected processed INT8 layout: '+name)
        k,n=w.shape
        if n==6144:
            args = (kernel.config, kernel.layer_param_names)
            replacement = (
                VerifyTransposeInt8Kernel(*args, row_major=verify_row_major) if verify_transpose
                else TritonInt8ScaledMMLinearKernel(*args)
            )
            pending.append((scheme,replacement,name,k,n))
    for scheme,replacement,name,k,n in pending:scheme.kernel=replacement
    counts={}
    for _,_,_,k,n in pending:counts['%dx%d'%(k,n)]=counts.get('%dx%d'%(k,n),0)+1
    print('MINIMAX_INT8_RUNTIME '+json.dumps({'backend':('qualified M128 transposed Lt; Core Triton otherwise'
        if verify_transpose else 'Core Triton for unqualified channelwise Lt shapes'),
        'geometries':counts,'weight_repack_count':0,'permanent_weight_mutations':0}),flush=True)
    return len(pending)
