# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 sliding-window layers: q/k norm + RoPE + q fp8 quant + KV cache write.

One kernel replaces what Inductor emits for these layers at decode (a combo
q-norm/rope/quant + k sum-sq kernel, a k norm/rope kernel) plus the separate
reshape_and_cache_flash launch. Semantics follow the compiled graph: the Gemma
RMSNorm (``x * rsqrt(mean(x^2) + eps) * (1 + w)``) and the neox RoPE are
evaluated in fp32 without intermediate rounding; q is quantized with the
attention layer's static scale; k is rounded to bf16 before the fp8 cache
store, as it was when it went through reshape_and_cache_flash.

Gated by VLLM_HCU_STEP5_SWA_FUSED_QK. The op takes the layer name, so it is a
graph splitting op (see CompilationConfig) like unified_kv_cache_update.

VLLM_HCU_STEP5_FUSED_GATE: the head-wise sigmoid gate and the per-token fp8
quantization of the o_proj input run as one kernel (step5_gate_quant_fp8, 3.4us
vs 8.1us for Inductor's three) and the fp8 o_proj takes the pre-quantized
input. On sliding layers that also take the fused q/k path, the epilogue is
folded into the split-KV decode combine instead (step5_swa_attention_gate_quant,
7.9us vs 15.7us for combine + gate/quant), a splitting op like the attention.
"""

import os

import torch

from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm_hcu.platforms import envs as henvs

SWA_FUSED_QK = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_SWA_FUSED_QK", "0") == "1"
)
FUSED_GATE = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FUSED_GATE", "0") == "1"
)
OP_NAME = "step5_swa_qk_norm_rope_cache"
GATED_OP_NAME = "step5_swa_attention_gate_quant"

# hb=8 / 1 warp: 4.05us at 64 tokens vs 12.46us for the three replaced kernels.
_HEADS_PER_PROG = 8


@triton.jit
def _norm_rope(x1, x2, w_ptr, cos, sin, eps, HALF: tl.constexpr, BH: tl.constexpr):
    offs = tl.arange(0, BH)
    m = offs < HALF
    ss = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    r = tl.rsqrt(ss / (2 * HALF) + eps)[:, None]
    w1 = tl.load(w_ptr + offs, mask=m, other=0.0) + 1.0
    w2 = tl.load(w_ptr + HALF + offs, mask=m, other=0.0) + 1.0
    n1 = x1 * r * w1[None, :]
    n2 = x2 * r * w2[None, :]
    return n1 * cos - n2 * sin, n2 * cos + n1 * sin


@triton.jit
def _swa_qk_norm_rope_cache_kernel(
    qkv_ptr,
    stride_t,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    cs_ptr,
    stride_cs,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    q_out_ptr,
    slot_ptr,
    num_slots,
    kc_ptr,
    vc_ptr,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    eps,
    PAGE: tl.constexpr,
    NQ: tl.constexpr,
    HB: tl.constexpr,
    HALF: tl.constexpr,
    BH: tl.constexpr,
):
    # grid = (tokens, NQ // HB + num_kv_heads): q head groups first, then one
    # program per kv head that also writes the cache.
    t = tl.program_id(0)
    g = tl.program_id(1)
    D: tl.constexpr = 2 * HALF
    offs = tl.arange(0, BH)
    dm = offs < HALF
    pos = tl.load(pos_ptr + t)
    cos = tl.load(cs_ptr + pos * stride_cs + offs, mask=dm, other=0.0).to(tl.float32)[
        None, :
    ]
    sin = tl.load(cs_ptr + pos * stride_cs + HALF + offs, mask=dm, other=0.0).to(
        tl.float32
    )[None, :]
    row = qkv_ptr + t.to(tl.int64) * stride_t
    if g < NQ // HB:
        h = g * HB + tl.arange(0, HB)
        base = row + h[:, None] * D + offs[None, :]
        x1 = tl.load(base, mask=dm[None, :], other=0.0).to(tl.float32)
        x2 = tl.load(base + HALF, mask=dm[None, :], other=0.0).to(tl.float32)
        o1, o2 = _norm_rope(x1, x2, qw_ptr, cos, sin, eps, HALF, BH)
        inv = 1.0 / tl.load(q_scale_ptr)
        o1 = tl.minimum(tl.maximum(o1 * inv, -448.0), 448.0).to(tl.float8e4nv)
        o2 = tl.minimum(tl.maximum(o2 * inv, -448.0), 448.0).to(tl.float8e4nv)
        ob = q_out_ptr + t.to(tl.int64) * (NQ * D) + h[:, None] * D + offs[None, :]
        tl.store(ob, o1, mask=dm[None, :])
        tl.store(ob + HALF, o2, mask=dm[None, :])
    else:
        kh = g - NQ // HB
        if t < num_slots:
            slot = tl.load(slot_ptr + t)
            if slot >= 0:
                kbase = row + (NQ + kh) * D + offs[None, :]
                x1 = tl.load(kbase, mask=dm[None, :], other=0.0).to(tl.float32)
                x2 = tl.load(kbase + HALF, mask=dm[None, :], other=0.0).to(tl.float32)
                o1, o2 = _norm_rope(x1, x2, kw_ptr, cos, sin, eps, HALF, BH)
                kinv = 1.0 / tl.load(k_scale_ptr)
                o1 = o1.to(tl.bfloat16).to(tl.float32) * kinv
                o2 = o2.to(tl.bfloat16).to(tl.float32) * kinv
                o1 = tl.minimum(tl.maximum(o1, -448.0), 448.0).to(tl.float8e4nv)
                o2 = tl.minimum(tl.maximum(o2, -448.0), 448.0).to(tl.float8e4nv)
                blk = slot // PAGE
                off = slot % PAGE
                kd = (
                    kc_ptr
                    + blk * stride_kb
                    + off * stride_ks
                    + kh * stride_kh
                    + offs[None, :]
                )
                tl.store(kd, o1, mask=dm[None, :])
                tl.store(kd + HALF, o2, mask=dm[None, :])
                num_kv = tl.num_programs(1) - NQ // HB
                vbase = row + (NQ + num_kv + kh) * D + offs[None, :]
                vinv = 1.0 / tl.load(v_scale_ptr)
                v1 = tl.load(vbase, mask=dm[None, :], other=0.0).to(tl.float32) * vinv
                v2 = (
                    tl.load(vbase + HALF, mask=dm[None, :], other=0.0).to(tl.float32)
                    * vinv
                )
                v1 = tl.minimum(tl.maximum(v1, -448.0), 448.0).to(tl.float8e4nv)
                v2 = tl.minimum(tl.maximum(v2, -448.0), 448.0).to(tl.float8e4nv)
                vd = (
                    vc_ptr
                    + blk * stride_vb
                    + off * stride_vs
                    + kh * stride_vh
                    + offs[None, :]
                )
                tl.store(vd, v1, mask=dm[None, :])
                tl.store(vd + HALF, v2, mask=dm[None, :])


def swa_qk_norm_rope_cache(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    q_out: torch.Tensor | None = None,
    heads_per_prog: int = _HEADS_PER_PROG,
    num_warps: int = 1,
) -> torch.Tensor:
    """Writes q as fp8 [tokens, num_heads * head_dim] into q_out (allocated if
    None) and returns it; writes k/v into the paged fp8 cache ([blocks, page,
    kv_heads, head_dim] views) at slot_mapping."""
    num_tokens = qkv.shape[0]
    head_dim = cos_sin_cache.shape[-1]
    if q_out is None:
        q_out = torch.empty(
            num_tokens,
            num_heads * head_dim,
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )
    if num_tokens == 0:
        return q_out
    if slot_mapping is None or key_cache is None:
        num_slots = 0
        slot_mapping = positions
        key_cache = value_cache = q_out.view(1, 1, 1, -1)
    else:
        num_slots = slot_mapping.shape[0]
    hb = heads_per_prog
    _swa_qk_norm_rope_cache_kernel[(num_tokens, num_heads // hb + num_kv_heads)](
        qkv,
        qkv.stride(0),
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_scale,
        k_scale,
        v_scale,
        q_out,
        slot_mapping,
        num_slots,
        key_cache,
        value_cache,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        eps,
        PAGE=key_cache.shape[1],
        NQ=num_heads,
        HB=hb,
        HALF=head_dim // 2,
        BH=triton.next_power_of_2(head_dim // 2),
        num_warps=num_warps,
    )
    return q_out


def _op_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    q_out: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    from vllm_hcu.v1.attention.backends.flash_attn import _split_kv_cache

    _, layer, kv_cache, slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    key_cache = value_cache = None
    if slot_mapping is not None and kv_cache.numel() > 0:
        head_dim = cos_sin_cache.shape[-1]
        key_cache, value_cache = _split_kv_cache(kv_cache, head_dim)
        key_cache = key_cache.view(torch.float8_e4m3fn)
        value_cache = value_cache.view(torch.float8_e4m3fn)
    else:
        slot_mapping = None
    swa_qk_norm_rope_cache(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        eps,
        layer._q_scale,
        layer._k_scale,
        layer._v_scale,
        num_heads,
        num_kv_heads,
        key_cache,
        value_cache,
        slot_mapping,
        q_out,
    )
    return qkv.new_empty(0)


def _op_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    q_out: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return qkv.new_empty(0)


# q_out is allocated by the caller: this is a splitting op, and a tensor it
# allocated itself would reach the next piecewise CUDA graph at an address
# that graph did not capture.
if SWA_FUSED_QK:
    direct_register_custom_op(
        op_name=OP_NAME,
        op_func=_op_impl,
        mutates_args=["q_out"],
        fake_impl=_op_fake,
    )


def _gate_quant_impl(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_hcu.v1.attention.backends.swa_decode import gate_quant_fp8

    xq = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    gate_quant_fp8(x, gate, xq, xs)
    return xq, xs


def _gate_quant_fake(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    return xq, xs


if FUSED_GATE:
    direct_register_custom_op(
        op_name="step5_gate_quant_fp8",
        op_func=_gate_quant_impl,
        mutates_args=[],
        fake_impl=_gate_quant_fake,
    )


def gate_quant(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x bf16 [T, NH * D] * sigmoid(gate [T, NH]) per head -> per-token fp8
    (xq [T, NH * D], xs fp32 [T, 1]) for Step5FP8LinearMethod."""
    return torch.ops.vllm.step5_gate_quant_fp8(x, gate)


def _gated_op_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep
    attn_metadata, layer, kv_cache, _ = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    if attn_metadata is None:
        # Profiling run.
        xq.zero_()
        xs.fill_(1.0)
        return
    if layer.impl.forward_swa_gate_quant(
        layer, query, kv_cache, attn_metadata, gate, xq, xs
    ):
        return
    from vllm_hcu.v1.attention.backends.swa_decode import gate_quant_fp8

    output = torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)
    layer.impl.forward(layer, query, key, value, kv_cache, attn_metadata, output=output)
    gate_quant_fp8(
        output.view(output.shape[0], -1),
        gate,
        xq,
        xs,
        num_valid=attn_metadata.num_actual_tokens,
    )


if SWA_FUSED_QK and FUSED_GATE:
    direct_register_custom_op(
        op_name=GATED_OP_NAME,
        op_func=_gated_op_impl,
        mutates_args=["xq", "xs"],
    )


def can_use(attn_module) -> bool:
    """attn_module: Step5Attention. Sliding layers on the HCU flash-attn backend
    with an fp8 cache and a full-width neox RoPE."""
    attn = attn_module.attn
    rope = attn_module.rotary_emb
    return (
        SWA_FUSED_QK
        and not attn_module.is_sparse_layer
        and attn.sliding_window is not None
        and attn.kv_cache_dtype in ("fp8", "fp8_e4m3")
        and attn.query_quant is not None
        and attn_module.use_rope
        and getattr(rope, "is_neox_style", False)
        and getattr(rope, "rotary_dim", -1) == attn_module.head_dim
        and rope.cos_sin_cache.shape[-1] == attn_module.head_dim
        and attn_module.head_dim % 2 == 0
        and attn_module.num_heads % _HEADS_PER_PROG == 0
        and type(attn.impl).__module__ == "vllm_hcu.v1.attention.backends.flash_attn"
    )


def forward(attn_module, positions, qkv, v_view, k_view, gate=None):
    """q/k/v from the fused op, then attention without the separate cache write.

    With ``gate`` (FUSED_GATE), returns the gated, fp8-quantized o_proj input
    (xq, xs) instead of the bf16 attention output."""
    attn = attn_module.attn
    encoded = _encode_layer_name(attn.layer_name)
    head_dim = attn_module.head_dim
    q = torch.empty(
        qkv.shape[0],
        attn_module.num_heads * head_dim,
        dtype=torch.float8_e4m3fn,
        device=qkv.device,
    )
    dummy = torch.ops.vllm.step5_swa_qk_norm_rope_cache(
        qkv,
        positions,
        attn_module.q_norm.weight,
        attn_module.k_norm.weight,
        attn_module.rotary_emb.cos_sin_cache,
        attn_module.q_norm.variance_epsilon,
        attn_module.num_heads,
        attn_module.num_kv_heads,
        q,
        encoded,
    )
    if gate is not None:
        num_tokens = qkv.shape[0]
        xq = torch.empty(
            num_tokens,
            attn_module.num_heads * head_dim,
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )
        xs = torch.empty(num_tokens, 1, dtype=torch.float32, device=qkv.device)
        torch.ops.vllm.step5_swa_attention_gate_quant(
            q.view(-1, attn_module.num_heads, head_dim),
            k_view.view(-1, attn_module.num_kv_heads, head_dim),
            v_view.view(-1, attn_module.num_kv_heads, head_dim),
            gate,
            xq,
            xs,
            encoded,
            kv_cache_dummy_dep=dummy,
        )
        return xq, xs
    output = torch.empty(
        qkv.shape[0],
        attn_module.num_heads * head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    torch.ops.vllm.unified_attention_with_output(
        q.view(-1, attn_module.num_heads, head_dim),
        k_view.view(-1, attn_module.num_kv_heads, head_dim),
        v_view.view(-1, attn_module.num_kv_heads, head_dim),
        output.view(-1, attn_module.num_heads, head_dim),
        encoded,
        kv_cache_dummy_dep=dummy,
    )
    return output
