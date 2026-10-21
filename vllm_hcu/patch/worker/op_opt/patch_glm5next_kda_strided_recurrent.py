# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid .contiguous() copies ahead of GLM5Next's KDA recurrent kernel.

The target implementation's ``fused_recurrent_kda`` unconditionally copies
q/k/v/g/beta to a dense layout before launching the fused recurrent kernel
(``fused_recurrent_gated_delta_rule_fwd_kernel``, hard-coded to a dense
``H*K``/``HV*V`` token stride). On the decode and spec-verify paths, q/k/v are
split out of the merged qkv ``causal_conv1d_update`` output and are therefore
non-contiguous, so a profiler trace shows 4-5 elementwise copy kernels between
``causal_conv1d_update`` and the recurrent kernel. sglang's equivalent HCU
kernel (``fused_sigmoid_gating_delta_rule_update_kernel_hcu``) takes per-token
strides and reads these inputs directly, with none of these intermediate
copies.

This patch swaps in an alternate ``fused_recurrent_kda`` that forwards
per-token strides into a (locally defined) copy of the Triton kernel and
skips the copy for any input whose (head, dim) layout is already dense.
Gated by VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS, enabled by default and
consulted once at patch-apply time. Set it to 0 to disable; when off the
target module is left untouched.

The Triton kernel below is copied from
``vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent`` /
``kernels`` as of vLLM 0.28.1+das (upstream flash-linear-attention, Songlin
Yang, Yu Zhang), with per-token stride parameters added and two upstream bugs
fixed (gate reading missing mask when K is not a power of 2, and final_state
allocation sized T instead of B*T when inplace_final_state=False). When K is
non-power-of-2 or inplace_final_state=False, outputs will not match the
(buggy) upstream kernel bit-for-bit. If the upstream kernel changes, this copy
must be re-synced manually; ``require_exact_signature`` below only guards the
Python wrapper signature, not the kernel body.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.glm5next.nvidia.kda"
PATCH_ID = "worker.op_opt.glm5next_kda_strided_recurrent"
TARGETS = (f"{TARGET_MODULE}.fused_recurrent_kda",)
_MARKER = "_vllm_hcu_glm5next_kda_strided_recurrent_applied"
_WRAPPER = "_vllm_hcu_glm5next_kda_strided_recurrent_wrapper"


def _inner_dense(x: torch.Tensor) -> bool:
    # [B, T, h, d] / [B, T, h]: everything after the token dim must be dense;
    # the token stride itself may be arbitrary.
    if x.stride(-1) != 1:
        return False
    if x.ndim == 4:
        return x.shape[2] == 1 or x.stride(2) == x.shape[3]
    return True


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    # B > 1 without cu_seqlens indexes tokens as i_n * T + t, which only
    # matches a strided view when the batch dim is packed; keep that rare
    # case on the copy path via the B == 1 requirement below.
    if x.shape[0] == 1 and _inner_dense(x):
        return x
    return x.contiguous()


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def _strided_fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    a_log,
    g_bias,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    # Per-token strides (in elements) of the inputs. Runtime args, not
    # constexpr. Triton still specializes integer args on ==1 / %16
    # alignment, so distinct stride classes compile separate variants; for a
    # given model the stride set is small and fixed (the conv-split and
    # contiguous layouts), so this is bounded at a handful of compiles.
    # Deliberately NOT in do_not_specialize: that changes codegen enough to
    # lose bit-exactness with the upstream kernel (~1 fp32 ulp).
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    stride_beta,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    SIGMOID_BETA: tl.constexpr,
    COMPUTE_GATE: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * stride_beta + i_hv * V + o_v
    else:
        p_beta = beta + bos * stride_beta + i_hv

    if not IS_KDA:
        p_g = g + bos * stride_g + i_hv
    else:
        p_gk = g + bos * stride_g + i_hv * K + o_k

    if COMPUTE_GATE:
        b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk, mask=mask_k, other=0.0).to(tl.float32)
            if COMPUTE_GATE:
                # Replicates kda_gate_fwd_kernel's SAFE_GATE branch
                # bit-for-bit: y = lb / (1 + exp(-exp(A)*(g+bias))).
                b_gk += tl.load(g_bias + i_h * K + o_k, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
            b_h *= exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        if SIGMOID_BETA:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_q
        p_k += stride_k
        p_o += HV * V
        p_v += stride_v
        if not IS_KDA:
            p_g += stride_g
        else:
            p_gk += stride_g
        p_beta += stride_beta


def _strided_fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    out: torch.Tensor | None,
    sigmoid_beta: bool,
    a_log: torch.Tensor | None,
    g_bias: torch.Tensor | None,
    compute_gate: bool,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.utils.math_utils import cdiv, next_power_of_2

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    if compute_gate:
        assert a_log is not None and g_bias is not None, (
            "compute_gate requires a_log and g_bias"
        )
        assert lower_bound is not None, (
            "compute_gate implements the bounded (safe_gate) branch only"
        )
        a_log = a_log.reshape(-1).contiguous()
        g_bias = g_bias.reshape(-1).contiguous()

    if out is None:
        o = torch.empty_like(k)
    else:
        assert out.shape == k.shape and out.dtype == k.dtype
        assert out.is_contiguous()
        o = out
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(B * T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    _strided_fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_q=q.stride(1),
        stride_k=k.stride(1),
        stride_v=v.stride(1),
        stride_g=g.stride(1),
        stride_beta=beta.stride(1),
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
        SIGMOID_BETA=sigmoid_beta,
        a_log=a_log,
        g_bias=g_bias,
        COMPUTE_GATE=compute_gate,
        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


def apply_to_module(module: ModuleType) -> bool:
    kda = load_exact_module(TARGET_MODULE, module)
    wrapped = ((kda, "fused_recurrent_kda", TARGETS[0], _WRAPPER),)
    if already_applied(kda, _MARKER, wrapped):
        return False
    original = require_callable(kda, "fused_recurrent_kda", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "q",
            "k",
            "v",
            "g",
            "beta",
            "scale",
            "initial_state",
            "inplace_final_state",
            "use_qk_l2norm_in_kernel",
            "cu_seqlens",
            "ssm_state_indices",
            "num_accepted_tokens",
            "out",
            "sigmoid_beta",
            "a_log",
            "g_bias",
            "compute_gate",
            "lower_bound",
        ),
        defaults={
            "beta": None,
            "scale": None,
            "initial_state": None,
            "inplace_final_state": True,
            "use_qk_l2norm_in_kernel": True,
            "cu_seqlens": None,
            "ssm_state_indices": None,
            "num_accepted_tokens": None,
            "out": None,
            "sigmoid_beta": False,
            "a_log": None,
            "g_bias": None,
            "compute_gate": False,
            "lower_bound": -5.0,
        },
        var_keyword="kwargs",
    )

    import vllm_hcu.platforms.envs as henvs

    if not henvs.VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS or not henvs.VLLM_HCU_USE_CUSTOM_OPS:
        # Leave the target untouched when the feature or its master switch is off.
        return False

    @functools.wraps(original)
    def hcu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta=None,
        scale=None,
        initial_state=None,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        out=None,
        sigmoid_beta=False,
        a_log=None,
        g_bias=None,
        compute_gate=False,
        lower_bound=-5.0,
        **kwargs,
    ):
        if kwargs:
            # Unknown extra arguments (e.g. added by a future vLLM); the
            # strided copy does not know how to honor them, so defer to the
            # original implementation instead of silently dropping them.
            return original(
                q, k, v, g,
                beta=beta, scale=scale, initial_state=initial_state,
                inplace_final_state=inplace_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens, out=out,
                sigmoid_beta=sigmoid_beta, a_log=a_log, g_bias=g_bias,
                compute_gate=compute_gate, lower_bound=lower_bound,
                **kwargs,
            )
        if cu_seqlens is not None and q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                "when using `cu_seqlens`. Please flatten variable-length "
                "inputs before processing."
            )
        if scale is None:
            scale = k.shape[-1] ** -0.5

        return _strided_fused_recurrent_kda_fwd(
            q=_maybe_contiguous(q),
            k=_maybe_contiguous(k),
            v=_maybe_contiguous(v),
            g=_maybe_contiguous(g),
            beta=_maybe_contiguous(beta),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            out=out,
            sigmoid_beta=sigmoid_beta,
            a_log=a_log,
            g_bias=g_bias,
            compute_gate=compute_gate,
            lower_bound=lower_bound,
        )

    setattr(hcu_fused_recurrent_kda, _WRAPPER, True)
    setattr(kda, "_vllm_hcu_original_fused_recurrent_kda", original)
    setattr(kda, "fused_recurrent_kda", hcu_fused_recurrent_kda)
    setattr(kda, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
