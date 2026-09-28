# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.fla.ops.op import exp, log
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2


@triton.heuristics(
    {
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit
def _kda_gate_beta_fwd_kernel(
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    gate,
    beta_out,
    lower_bound,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    T,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D

    p_g = raw_g + o_t[:, None] * stride_g_token + i_h * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(
            dt_bias + i_h * D + o_d,
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        b_g += b_bias[None, :]

    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_scaled = b_g * softplus_beta
        b_softplus = tl.where(
            b_scaled > softplus_threshold,
            b_g,
            log(1.0 + tl.exp(b_scaled)) / softplus_beta,
        )
        b_gate = -b_a * b_softplus

    p_gate = gate + (o_t[:, None] * H + i_h) * D + o_d[None, :]
    tl.store(
        p_gate,
        b_gate,
        mask=m_t[:, None] & m_d[None, :],
    )

    b_beta = tl.load(
        raw_beta + o_t * stride_beta_token + i_h,
        mask=m_t,
        other=0.0,
    ).to(tl.float32)
    tl.store(beta_out + o_t * H + i_h, tl.sigmoid(b_beta), mask=m_t)


def _fused_kda_gate_beta(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = raw_g.shape
    assert B == 1
    assert raw_beta.shape == (B, T, H)
    assert raw_g.stride()[2:] == (D, 1)
    assert raw_beta.stride(2) == 1
    gate = torch.empty((B, T, H, D), dtype=torch.float32, device=raw_g.device)
    beta = torch.empty((B, T, H), dtype=torch.float32, device=raw_beta.device)

    BT = 16
    _kda_gate_beta_fwd_kernel[(cdiv(T, BT), H)](
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        gate=gate,
        beta_out=beta,
        lower_bound=lower_bound,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        T=T,
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        H=H,
        D=D,
        BT=BT,
        BD=next_power_of_2(D),
        num_warps=4,
    )
    return gate, beta


@triton.heuristics(
    {
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    out,
    state,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    i_v = pid % tl.cdiv(V, BV)
    i_nh = pid // tl.cdiv(V, BV)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    sequence_length = eos - bos
    if sequence_length == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_state = m_v[:, None] & m_k[None, :]

    if IS_SPEC_DECODING:
        initial_token = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    else:
        initial_token = 0
    state_index = tl.load(state_indices + i_n * stride_indices_seq + initial_token).to(
        tl.int64
    )
    p_out = out + bos * stride_out_token + i_h * V + o_v
    if state_index <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=m_v)
        return

    p_state = (
        state
        + state_index * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_state = tl.load(p_state, mask=m_state, other=0.0).to(tl.float32)

    p_q = q + bos * stride_qkv_token + i_h * K + o_k
    p_k = k + bos * stride_qkv_token + i_h * K + o_k
    p_v = v + bos * stride_qkv_token + i_h * V + o_v
    p_g = g + bos * stride_g_token + i_h * K + o_k
    p_beta = beta + bos * stride_beta_token + i_h
    for i_t in tl.range(0, sequence_length, num_stages=num_stages):
        b_q = tl.load(p_q, mask=m_k, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=m_k, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=m_v, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale

        b_gate = tl.load(
            p_g,
            mask=m_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                b_bias = tl.load(
                    dt_bias + i_h * K + o_k,
                    mask=m_k,
                    other=0.0,
                ).to(tl.float32)
                b_gate += b_bias
            b_a = exp(tl.load(A_log + i_h).to(tl.float32))
            if USE_LOWER_BOUND:
                b_gate = lower_bound * tl.sigmoid(b_a * b_gate)
            else:
                b_softplus = tl.where(
                    b_gate > 20.0,
                    b_gate,
                    log(1.0 + tl.exp(b_gate)),
                )
                b_gate = -b_a * b_softplus

        b_state *= exp(b_gate[None, :])
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]
        b_out = tl.sum(b_state * b_q[None, :], axis=1)
        tl.store(
            p_out,
            b_out.to(p_out.dtype.element_ty),
            mask=m_v,
            eviction_policy="evict_first",
        )

        final_state_index = tl.load(state_indices + i_n * stride_indices_seq + i_t).to(
            tl.int64
        )
        if final_state_index > 0:
            p_final_state = (
                state
                + final_state_index * stride_state_token
                + i_h * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_final_state,
                b_state.to(p_final_state.dtype.element_ty),
                mask=m_state,
            )

        p_q += stride_qkv_token
        p_k += stride_qkv_token
        p_v += stride_qkv_token
        p_g += stride_g_token
        p_beta += stride_beta_token
        p_out += stride_out_token


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch recurrent KDA with dense inner dimensions and row strides."""
    # Delegate the recurrent kernel to boltops (static config, no autotune).
    from boltops.fla.kda import fused_recurrent_kda_fwd as impl

    return impl(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        out=out,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA from raw gate and beta inputs.

    This vLLM wrapper applies the gate activation and beta sigmoid, selecting
    whether to materialize them before launching the recurrent kernel.
    """
    # Delegate to boltops, which carries the identical fused-gate logic.
    from boltops.fla.kda import fused_recurrent_kda as impl

    return impl(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        fuse_gate=fuse_gate,
    )


@triton.jit
def fused_recurrent_kda_packed_decode_kernel(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_state = state + state_idx * stride_state_token
    p_state += i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    # Q, K, and V occupy consecutive channel ranges, while the token stride
    # may also include the output-gate projection that follows packed QKV.
    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = raw_g + i_n * stride_g_token + i_h * K + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_g += b_bias
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_softplus = tl.where(
            b_g > SOFTPLUS_THRESHOLD,
            b_g,
            log(1.0 + tl.exp(b_g)),
        )
        b_gate = -b_a * b_softplus

    b_state *= exp(b_gate[None, :])
    b_v -= tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )
    b_v *= b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-token KDA decode directly from packed post-conv QKV."""
    # Delegate to boltops (static config, no autotune).
    from boltops.fla.kda import fused_recurrent_kda_packed_decode as impl

    return impl(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        state_indices,
        scale=scale,
    )
