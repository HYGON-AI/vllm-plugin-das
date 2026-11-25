"""MiniMax-M3.1 checkpoint contract over BoltOPs' existing Triton W4A8 experts.

Checkpoint: signed two's-complement INT4, even K in high nibble, INT8
containers, FP32 per-output scales in the INT8 domain (nibble * 16 * scale).
The loader fuses w1(gate),w3(up) as split halves. BoltOPs swigluoai actually
reads alternating gate/up rows, despite its public docstring's split claim.
Prepare that layout once, before caching processed weights. Restored processed
IPC weights must NOT be transformed a second time. w2 stays unchanged.

GEMM1 stores BF16/FP16, OAI activation clamps gate above +limit and up to
[-limit,+limit], computes gate*sigmoid(alpha*gate)*(up+1), then stores the
same dtype before dynamic token INT8 requantization. GEMM2 applies routing
before its output dtype store and the top-k sum. The model owns shared experts
and its outer routed_scaling_factor=2; this entry applies neither.

No framework import or plugin initialization occurs here. Production caller:
hcu_vllm_das.quantization.minimax31 via the registered quant method.
"""
import math
import torch


def prepare_minimax31_w13(w13, w13_scale):
    """Convert split gate/up to alternating rows once; preserve raw scales."""
    if w13.ndim != 3 or w13.shape[1] % 2 or w13.dtype != torch.int8:
        raise ValueError('w13 must be packed INT8 [E,2I,H/2]')
    e, n, k = w13.shape
    if w13_scale.dtype != torch.float32 or w13_scale.shape not in ((e,n),(e,n,1)):
        raise ValueError('w13_scale must be FP32 [E,2I] or [E,2I,1]')
    if w13_scale.device != w13.device:
        raise ValueError('Weight and scale devices differ')
    weight = w13.reshape(e,2,n//2,k).transpose(1,2).contiguous().reshape(e,n,k)
    scale = w13_scale.reshape(e,2,n//2,-1).transpose(1,2).contiguous().reshape(w13_scale.shape)
    return weight, scale


def apply_minimax31_w4a8(x, w13, w2, w13_scale, w2_scale,
                         topk_weights, topk_ids, *, alpha=1.702, beta=1.0,
                         limit=7.0, expert_map=None, w13_layout, inplace=False):
    """Run routed experts; input weights must be prepared interleaved INT4.

    Canonical checkpoint scales are accepted without pre-multiplication. The
    exact factor 16 converts them to BoltOPs' signed-nibble dequant domain.
    No disk fallback, dequantized BF16 weights, or alternate activation path.
    """
    if w13_layout != 'interleaved':
        raise ValueError('Prepare w13 once with prepare_minimax31_w13 before apply')
    if beta != 1.0 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError('OAI experts require beta=1, positive finite alpha/limit')
    if x.ndim != 2 or x.dtype not in (torch.bfloat16,torch.float16) or not x.is_cuda or not x.is_contiguous():
        raise ValueError('x must be contiguous CUDA/HIP BF16 or FP16 [M,H]')
    if w13.ndim != 3 or w2.ndim != 3 or w13.dtype != torch.int8 or w2.dtype != torch.int8:
        raise ValueError('Weights must be packed INT8 [E,N,K/2]')
    e,n,k=w13.shape
    if n%2 or 2*k!=x.shape[1] or w2.shape!=(e,x.shape[1],n//4):
        raise ValueError('Packed expert dimensions disagree')
    for name,scale,rows in (('w13',w13_scale,n),('w2',w2_scale,x.shape[1])):
        if scale.dtype!=torch.float32 or scale.shape not in ((e,rows),(e,rows,1)):
            raise ValueError(name+' scales must be FP32 per-output-channel')
    if topk_weights.shape != topk_ids.shape or topk_ids.ndim!=2 or topk_ids.shape[0]!=x.shape[0]:
        raise ValueError('Routing tensors must match [M,top_k]')
    if topk_weights.dtype!=torch.float32 or topk_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('Routing requires FP32 weights and INT32/INT64 IDs')
    tensors=(w13,w2,w13_scale,w2_scale,topk_weights,topk_ids)
    if any(t.device!=x.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Expert inputs must be contiguous on the same device')
    if expert_map is not None and (expert_map.device!=x.device or expert_map.dtype!=torch.int32):
        raise ValueError('expert_map must be local INT32 global-to-local indices')
    if not isinstance(inplace,bool) or (inplace and (expert_map is not None or topk_ids.shape[1] != 1)):
        raise ValueError('inplace reuse requires owned packed local topk1 input without expert_map')
    if x.shape[0]==0:return torch.empty_like(x)
    from boltops.fused_moe import fused_experts
    return fused_experts(x,w13,w2,topk_weights,topk_ids,inplace=inplace,
        activation='swigluoai',is_gated=True,apply_router_weight_on_input=False,
        use_int4_w4a8=True,per_channel_quant=True,
        w1_scale=w13_scale*16.0,w2_scale=w2_scale*16.0,
        expert_map=expert_map,gemm1_alpha=alpha,gemm1_limit=limit,
        routed_scaling_factor=1.0,output_dtype=x.dtype)
