# SPDX-License-Identifier: Apache-2.0
"""Plugin-owned adapters for aiter's contiguous, signed W4A8 GEMMs.

Weights store even K in the high nibble and odd K in the low nibble.
Scales are dequantization multipliers per output channel. No weight
unpacking, layout shuffle, or new GPU kernel is used during inference.
"""
import torch


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Portable numerical-reference unpacker; not used during inference."""
    pair = torch.stack(((packed >> 4) & 15, packed & 15), dim=-1)
    return torch.where(pair >= 8, pair - 16, pair).to(torch.int8).flatten(-2)


def linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
           bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply [N,K/2] INT4 weight to [M,K], quantizing each input row to INT8."""
    from aiter import per_token_quant_triton
    from aiter.ops.triton.moe_op import fused_moe
    import triton.language as tl

    if x.ndim != 2 or packed.ndim != 2 or x.shape[1] != 2 * packed.shape[1]:
        raise ValueError('W4A8 linear requires x[M,K], packed[N,K/2]')
    if scales.shape != (packed.shape[0], 1):
        raise ValueError('W4A8 linear requires channel scales[N,1]')
    m, n = x.shape[0], packed.shape[0]
    output = torch.empty((m,n),dtype=x.dtype,device=x.device)
    if m == 0:
        return output
    q, xscale = per_token_quant_triton(x.contiguous(), quant_dtype=torch.int8)
    block_m = 16
    padded = (m + block_m - 1) // block_m * block_m
    # A single expert needs no sorting: padding IDs are >= the token count.
    sorted_ids = torch.arange(padded,device=x.device,dtype=torch.int32)
    expert_ids = torch.zeros((padded//block_m,),device=x.device,dtype=torch.int32)
    token_count = torch.full((1,),padded,device=x.device,dtype=torch.int32)
    ids = torch.zeros((m,1),device=x.device,dtype=torch.int32)
    compute_type = {torch.bfloat16:tl.bfloat16,torch.float16:tl.float16,
                    torch.float32:tl.float32}[x.dtype]
    fused_moe(q, packed.unsqueeze(0), output, xscale, scales.unsqueeze(0),
              None, None, ids, sorted_ids, None, expert_ids, token_count,
              False, 1, compute_type, use_int4_w4a8=True,
              per_channel_quant=True,
              config={'BLOCK_SIZE_M':block_m,'BLOCK_SIZE_N':64,
                      'BLOCK_SIZE_K':128,'GROUP_SIZE_M':1,
                      'num_warps':4,'num_stages':2})
    if bias is not None:
        output.add_(bias)
    return output


def moe(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
        s13: torch.Tensor, s2: torch.Tensor, topk_weights: torch.Tensor,
        topk_ids: torch.Tensor, *, gemm1_limit: float = 10.,
        expert_map: torch.Tensor | None = None, global_num_experts: int = -1,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: float = 1.) -> torch.Tensor:
    """Apply routed W4A8 experts; w13 rows are [gate, up].

    aiter applies SiLU(min(gate, limit)) * clamp(up, -limit, limit),
    then dynamically quantizes the intermediate activation per token.
    """
    from aiter.ops.triton.fused_moe import fused_experts_impl

    return fused_experts_impl(
        x.contiguous(), w13, w2, topk_weights, topk_ids,
        output_dtype=x.dtype, use_int4_w4a8=True, per_channel_quant=True,
        w1_scale=s13, w2_scale=s2, activation='silu', is_gated=True,
        gemm1_limit=gemm1_limit, expert_map=expert_map,
        global_num_experts=global_num_experts,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor)
