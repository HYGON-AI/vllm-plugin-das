# SPDX-License-Identifier: Apache-2.0
"""Opt-in Step5 DSA Q/K RMSNorm and partial NeoX RoPE, retaining BF16 Q/K.

This does not write KV, touch the indexer, switch streams or copy V. It matches
native BF16 RoPE: normalized values and each rotary product round to BF16.
All tensor dispatch stays inside the opaque op so a prefill trace cannot freeze
decode onto the fallback. Unsupported fast layouts retain the original calls.
"""
import math
import os

import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.platforms import envs as henvs

_ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_DSA_FUSED_QK", "0") == "1"
)
_qk_kernel = None
if _ENABLED:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _step5_dsa_qk_native_bf16_kernel(
        X, Q, K, QW, KW, CS, POS,
        SX: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, SC: tl.constexpr,
        NQ: tl.constexpr, EPS: tl.constexpr,
    ):
        token, head = tl.program_id(0), tl.program_id(1)
        is_q = head < NQ
        col = tl.arange(0, 256)
        valid = col < 192
        base = token.to(tl.int64) * SX + head * 192
        x = tl.load(X + base + col, mask=valid, other=0.).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / 192.0 + EPS)
        qw = tl.load(QW + col, mask=valid & is_q, other=0.).to(tl.float32)
        kw = tl.load(KW + col, mask=valid & ~is_q, other=0.).to(tl.float32)
        weight = tl.where(is_q, qw, kw) + 1.0
        normalized = ((x * inverse_rms) * weight).to(tl.bfloat16)

        rotary = col < 64
        partner_col = tl.where(col < 32, col + 32, col - 32)
        partner = tl.load(X + base + partner_col, mask=rotary, other=0.).to(tl.float32)
        pqw = tl.load(QW + partner_col, mask=rotary & is_q, other=0.).to(tl.float32)
        pkw = tl.load(KW + partner_col, mask=rotary & ~is_q, other=0.).to(tl.float32)
        partner_weight = tl.where(is_q, pqw, pkw) + 1.0
        partner = ((partner * inverse_rms) * partner_weight).to(tl.bfloat16)
        position = tl.load(POS + token).to(tl.int64)
        pair = col % 32
        cos = tl.load(CS + position * SC + pair, mask=rotary, other=0.).to(tl.float32)
        sin = tl.load(CS + position * SC + 32 + pair, mask=rotary, other=0.).to(tl.float32)
        # Native ApplyRotaryEmb uses BF16 multiplication outputs, then BF16
        # add/subtract. Do not contract these products into an FP32 FMA.
        product = (normalized.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
        partner_product = (partner.to(tl.float32) * sin).to(tl.bfloat16).to(tl.float32)
        rotated = tl.where(col < 32, product - partner_product,
                           product + partner_product).to(tl.bfloat16)
        output = tl.where(rotary, rotated, normalized)
        tl.store(Q + token.to(tl.int64) * SQ + head * 192 + col,
                 output, mask=valid & is_q)
        tl.store(K + token.to(tl.int64) * SK + (head - NQ) * 192 + col,
                 output, mask=valid & ~is_q)

    _qk_kernel = _step5_dsa_qk_native_bf16_kernel


def _inputs_eligible(qkv, positions, q_weight, k_weight, cache, num_heads, num_kv_heads):
    tensors = (qkv, positions, q_weight, k_weight, cache)
    return (
        all(isinstance(t, torch.Tensor) for t in tensors)
        and num_heads > 0 and num_kv_heads > 0
        and qkv.is_cuda and qkv.ndim == 2 and qkv.dtype == torch.bfloat16
        and qkv.shape[1] == (num_heads + 2 * num_kv_heads) * 192
        and qkv.stride(1) == 1
        and positions.ndim == 1 and positions.shape[0] == qkv.shape[0]
        and positions.dtype in (torch.int32, torch.int64) and positions.stride(0) == 1
        and cache.ndim == 2 and cache.shape[0] > 0 and cache.shape[1] == 64
        and cache.dtype == torch.bfloat16 and cache.stride(1) == 1
        and all(w.shape == (192,) and w.dtype == torch.float32 and w.stride(0) == 1
                for w in (q_weight, k_weight))
        and all(t.device == qkv.device for t in tensors[1:])
    )


def can_dispatch(attn_module):
    """Select the opaque op using module configuration only, never Tensor guards."""
    if not (
        henvs.optional_custom_op_enabled(_ENABLED)
        and henvs.optional_custom_op_enabled(henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM)
        and getattr(attn_module, "_hcu_step5_owner", False)
        and getattr(attn_module, "is_sparse_layer", False)
        and getattr(attn_module, "head_dim", None) == 192
        and getattr(attn_module, "use_rope", False)
        and getattr(attn_module, "num_heads", 0) > 0
        and getattr(attn_module, "num_kv_heads", 0) > 0
    ):
        return False
    rope = attn_module.rotary_emb
    if (not getattr(rope, "is_neox_style", False)
            or getattr(rope, "rotary_dim", None) != 64
            or getattr(rope, "head_size", None) != 192):
        return False
    if getattr(getattr(rope, "_forward_method", None), "__name__", None) != "forward_native":
        return False
    for norm in (attn_module.q_norm, attn_module.k_norm):
        if (not getattr(norm, "_hcu_step5_owner", False)
                or getattr(getattr(norm, "_forward_method", None), "__name__", None) != "forward_hip"):
            return False
    eps = attn_module.q_norm.variance_epsilon
    return (
        eps == attn_module.k_norm.variance_epsilon and math.isfinite(eps) and eps > 0
    )


def _reference_norm(x, weight, eps):
    # Match HcuGemmaRMSNorm.forward_hip, including its unsupported-dtype/native
    # branch. The caller has already made the per-head input contiguous.
    if (henvs.optional_custom_op_enabled(henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM)
            and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
            and weight.dtype in (torch.bfloat16, torch.float32)):
        return torch.ops.vllm.hcu_step5_gemma_rms_norm(x, weight, eps)
    from vllm import ir

    return ir.ops.rms_norm(x, weight.float() + 1.0, eps)


def _reference(qkv, positions, q_weight, k_weight, cos_sin_cache,
               num_heads, num_kv_heads, eps):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rows = qkv.shape[0]
    q_size, k_size = num_heads * 192, num_kv_heads * 192
    # Native RoPE reshapes with an inferred head count, which is ambiguous
    # for zero tokens. Preserve the op's empty-output contract on fallback.
    if rows == 0:
        return (
            torch.empty((0, q_size), dtype=qkv.dtype, device=qkv.device),
            torch.empty((0, k_size), dtype=qkv.dtype, device=qkv.device),
        )
    q, k, _ = qkv.split((q_size, k_size, k_size), dim=-1)
    q = _reference_norm(q.view(rows, num_heads, 192).contiguous(), q_weight, eps).view(rows, q_size)
    k = _reference_norm(k.view(rows, num_kv_heads, 192).contiguous(), k_weight, eps).view(rows, k_size)
    if cos_sin_cache.device != qkv.device or cos_sin_cache.dtype != qkv.dtype:
        cos_sin_cache = cos_sin_cache.to(device=qkv.device, dtype=qkv.dtype)
    return RotaryEmbedding.forward_static(positions, q, k, 192, 64, cos_sin_cache, True)


def _impl(qkv: torch.Tensor, positions: torch.Tensor, q_weight: torch.Tensor,
          k_weight: torch.Tensor, cos_sin_cache: torch.Tensor,
          num_heads: int, num_kv_heads: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    # The model caches can_dispatch at startup. Recheck the master switch
    # inside the opaque op so later calls can still select the reference path.
    if (not henvs.optional_custom_op_enabled(_ENABLED)
            or _qk_kernel is None or qkv.shape[0] > 64
            or not _inputs_eligible(qkv, positions, q_weight, k_weight, cos_sin_cache, num_heads, num_kv_heads)):
        return _reference(qkv, positions, q_weight, k_weight, cos_sin_cache,
                          num_heads, num_kv_heads, eps)
    q = torch.empty((qkv.shape[0], num_heads * 192), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((qkv.shape[0], num_kv_heads * 192), dtype=qkv.dtype, device=qkv.device)
    if qkv.shape[0]:
        _qk_kernel[(qkv.shape[0], num_heads + num_kv_heads)](
            qkv, q, k, q_weight, k_weight, cos_sin_cache, positions,
            qkv.stride(0), q.stride(0), k.stride(0), cos_sin_cache.stride(0),
            NQ=num_heads, EPS=float(eps), num_warps=4, enable_fp_fusion=False,
        )
    return q, k


def _fake(qkv: torch.Tensor, positions: torch.Tensor, q_weight: torch.Tensor,
          k_weight: torch.Tensor, cos_sin_cache: torch.Tensor,
          num_heads: int, num_kv_heads: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((qkv.shape[0], num_heads * 192), dtype=qkv.dtype, device=qkv.device),
        torch.empty((qkv.shape[0], num_kv_heads * 192), dtype=qkv.dtype, device=qkv.device),
    )


direct_register_custom_op(op_name="step5_dsa_qk_norm_rope_bf16", op_func=_impl,
                          mutates_args=[], fake_impl=_fake)


def forward(attn_module, positions, qkv):
    return torch.ops.vllm.step5_dsa_qk_norm_rope_bf16(
        qkv, positions, attn_module.q_norm.weight, attn_module.k_norm.weight,
        attn_module.rotary_emb.cos_sin_cache, attn_module.num_heads,
        attn_module.num_kv_heads, attn_module.q_norm.variance_epsilon,
    )
