# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fused Qwen3.5 / Qwen3-Next sparse-MoE block for the HCU W8A16 AITER path.

The stock block issues ~16 kernels per layer: router GEMM, shared gate_up
GEMM, SiLU-mul, shared down GEMM, shared-expert-gate GEMM, sigmoid-mul,
top-k softmax, align/sort, routed UP, GLU, routed DOWN, moe_sum, a copy and
the shared+routed add.  On HCU every kernel costs >= ~5 us even inside a CUDA
graph, so at decode batch sizes the launch floor dominates.

This path keeps the math (and the fp16 shared expert) unchanged but:
  * concatenates [shared gate_up | router | shared_expert_gate] into one
    weight so a single GEMM produces all three activations;
  * reads gate/up and router logits in place (strided) with small Triton
    kernels instead of materialising contiguous copies;
  * folds moe_sum + sigmoid(shared_gate) * shared + add into one tail kernel.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_W8A16_BLOCK_M = 16


@triton.jit
def _strided_silu_mul_kernel(y_ptr, ldy, out_ptr, D: tl.constexpr,
                             BLOCK: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row = y_ptr + m.to(tl.int64) * ldy
    g = tl.load(row + offs).to(tl.float32)
    u = tl.load(row + D + offs).to(tl.float32)
    o = g / (1.0 + tl.exp(-g)) * u
    tl.store(out_ptr + m.to(tl.int64) * D + offs,
             o.to(out_ptr.dtype.element_ty))


@triton.jit
def _strided_topk_softmax_kernel(y_ptr, ldy, col0, w_ptr, id_ptr,
                                 E: tl.constexpr, TOPK: tl.constexpr,
                                 RENORM: tl.constexpr):
    m = tl.program_id(0)
    e = tl.arange(0, E)
    logits = tl.load(y_ptr + m.to(tl.int64) * ldy + col0 + e).to(tl.float32)
    # NaN rows (never produced by real tokens) must still yield in-range ids.
    logits = tl.where(logits == logits, logits, -float("inf"))
    row_max = tl.max(logits, 0)
    denom = tl.sum(tl.exp(logits - row_max), 0)
    ks = tl.arange(0, TOPK)
    ids = tl.zeros([TOPK], dtype=tl.int32)
    vals = tl.zeros([TOPK], dtype=tl.float32)
    work = logits
    for i in tl.static_range(TOPK):
        v = tl.max(work, 0)
        # lowest index among ties, matching the reference top-k kernel
        idx = tl.min(tl.where(work == v, e, E - 1), 0)
        ids = tl.where(ks == i, idx, ids)
        vals = tl.where(ks == i, v, vals)
        work = tl.where(e == idx, -float("inf"), work)
    p = tl.exp(vals - row_max)
    if RENORM:
        w = p / tl.sum(p, 0)
    else:
        w = p / denom
    tl.store(w_ptr + m * TOPK + ks, w)
    tl.store(id_ptr + m * TOPK + ks, ids)


@triton.jit
def _moe_tail_kernel(c3_ptr, sh_ptr, y_ptr, ldy, gate_col, out_ptr,
                     N: tl.constexpr, TOPK: tl.constexpr,
                     BLOCK: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m64 = m.to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(c3_ptr + (m64 * TOPK + k) * N + offs).to(tl.float32)
    g = tl.load(y_ptr + m64 * ldy + gate_col).to(tl.float32)
    s = tl.load(sh_ptr + m64 * N + offs).to(tl.float32)
    acc += s / (1.0 + tl.exp(-g))
    tl.store(out_ptr + m64 * N + offs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _moe_tail_add_rmsnorm_kernel(c3_ptr, sh_ptr, y_ptr, ldy, gate_col,
                                 res_ptr, w_ptr, out_ptr, eps,
                                 N: tl.constexpr, TOPK: tl.constexpr):
    """MoE tail fused with the following Gemma fused-add RMSNorm.

    moe = sum_k c3[m, k] + sigmoid(gate[m]) * shared[m]   (rounded to the
    activation dtype, as the stock block stores it); residual += moe in place;
    out = rmsnorm(residual) * (1 + w).
    """
    m64 = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    acc = tl.zeros([N], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(c3_ptr + (m64 * TOPK + k) * N + offs).to(tl.float32)
    g = tl.load(y_ptr + m64 * ldy + gate_col).to(tl.float32)
    s = tl.load(sh_ptr + m64 * N + offs).to(tl.float32)
    dt = out_ptr.dtype.element_ty
    moe = (acc + s / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
    z = moe + tl.load(res_ptr + m64 * N + offs).to(tl.float32)
    tl.store(res_ptr + m64 * N + offs, z.to(dt))
    rstd = tl.rsqrt(tl.sum(z * z, axis=0) / N + eps)
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(out_ptr + m64 * N + offs, (z * rstd * (1.0 + w)).to(dt))


class _FusedState:
    """Per-layer constants resolved once after weights are loaded."""

    __slots__ = ("w_cat", "ldy", "inter", "num_experts", "top_k",
                 "renormalize", "w13", "w2", "w13_scale", "w2_scale",
                 "modes")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _is_unquantized_linear(layer) -> bool:
    qm = getattr(layer, "quant_method", None)
    return type(qm).__name__ == "UnquantizedLinearMethod" and getattr(
        layer, "bias", None) is None


def _unsupported_reason(block) -> str | None:
    if getattr(block, "replicate_shared_expert", False):
        return "replicated shared expert"
    if getattr(block, "is_sequence_parallel", False):
        return "sequence parallel"
    if getattr(block, "enable_eplb", False):
        return "EPLB"
    if getattr(block, "tp_size", 1) != 1 or getattr(block, "ep_size", 1) != 1:
        return "tensor/expert parallel"
    shared = getattr(block, "shared_expert", None)
    if shared is None or getattr(shared, "expert_gate", None) is None:
        return "no gated shared expert"
    runner = block.experts
    router = getattr(runner, "router", None)
    routed = getattr(runner, "routed_experts", None)
    if router is None or routed is None:
        return "unknown MoE runner layout"
    if getattr(router, "scoring_func", "softmax") != "softmax":
        return "non-softmax routing"
    if getattr(router, "e_score_correction_bias", None) is not None:
        return "routing bias"
    if getattr(router, "routed_scaling_factor", 1.0) not in (None, 1.0):
        return "routed scaling"
    if getattr(router, "custom_routing_function", None) is not None:
        return "custom routing"
    if getattr(routed, "expert_map", None) is not None:
        return "expert map"
    if getattr(routed, "apply_router_weight_on_input", False):
        return "router weight on input"
    act = getattr(routed, "activation", "silu")
    act = getattr(act, "value", act)
    if str(act).lower() != "silu":
        return f"activation {act}"
    qm = getattr(routed, "quant_method", None)
    if type(qm).__name__ != "ModelOptW8A16MoEMethod":
        return f"quant method {type(qm).__name__}"
    from vllm_hcu.model_executor.layers.quantization import (
        modelopt_w8a16_hcu as w8a16)
    import vllm.envs as envs
    if not (w8a16._HAS_AITER and envs.VLLM_ROCM_USE_AITER_MOE):
        return "AITER W8A16 MoE disabled"
    for lin in (block.gate, block.shared_expert_gate, shared.gate_up_proj):
        if not _is_unquantized_linear(lin):
            return "quantized router/shared gate_up"
    if block.gate.weight.dtype not in (torch.float16, torch.bfloat16):
        return f"router dtype {block.gate.weight.dtype}"
    return None


def _down_mode(m: int, table_mode: int) -> int:
    """GEMM2 (down) mode for the W8A16 marlin kernel.

    aiter's gfx936 table pins mode 2 for every M.  A sweep on the
    E=256/N=512/K=2048 shape (bench/tune_moe_w8a16.py) finds mode 4 faster
    for M<=9 and mode 5 faster above (1.7x at M>=512); outputs match mode 2
    to fp16 accumulation-order noise.  Other table modes are left as is.
    """
    if table_mode != 2:
        return table_mode
    return 4 if m <= 9 else 5


def _build_state(block) -> _FusedState | None:
    reason = _unsupported_reason(block)
    if reason is not None:
        logger.warning_once(
            "Qwen3.5 fused MoE block disabled (%s); using the stock path.",
            reason)
        return None
    shared = block.shared_expert
    runner = block.experts
    routed = runner.routed_experts
    gu = shared.gate_up_proj.weight
    rt = block.gate.weight
    sg = block.shared_expert_gate.weight
    if not (gu.dtype == rt.dtype == sg.dtype):
        logger.warning_once("Qwen3.5 fused MoE block disabled (mixed dtypes).")
        return None
    inter2, hidden = gu.shape
    num_experts = rt.shape[0]
    n = inter2 + num_experts + 1
    # rocBLAS picks a split-K kernel for odd N at decode sizes (e.g. 1288 ->
    # 23 us at M=1); a 256-aligned N (1536) stays on the ~7 us GEMV tile.
    ldy = (n + 255) // 256 * 256
    w_cat = torch.zeros(ldy, hidden, dtype=gu.dtype, device=gu.device)
    w_cat[:inter2].copy_(gu.data)
    w_cat[inter2:inter2 + num_experts].copy_(rt.data)
    w_cat[inter2 + num_experts].copy_(sg.data[0])
    # Re-point the originals at the concatenated storage so the fused weight
    # does not cost extra memory and the stock path keeps working.
    shared.gate_up_proj.weight.data = w_cat[:inter2]
    block.gate.weight.data = w_cat[inter2:inter2 + num_experts]
    block.shared_expert_gate.weight.data = w_cat[inter2 + num_experts:
                                                 inter2 + num_experts + 1]
    qcfg = routed.quant_method.moe_quant_config
    from aiter.fused_moe_c import try_get_optimal_moe_config_marlin
    w13, w2 = routed.w13_weight, routed.w2_weight
    top_k = runner.router.top_k
    indtype = str(gu.dtype)
    # The down-mode sweep was run on Qwen3.5-35B-A3B's expert shape only.
    tuned_shape = (num_experts, inter2 // 2, hidden) == (256, 512, 2048)

    @functools.lru_cache(maxsize=None)
    def modes(m: int) -> tuple[int, int]:
        up = try_get_optimal_moe_config_marlin(
            w13.shape, w2.shape, top_k, "int8_w8a16", m,
            is_bottom=False, use_moe_wna16_cuda=True, indtype=indtype)
        down = try_get_optimal_moe_config_marlin(
            w13.shape, w2.shape, top_k, "int8_w8a16", m,
            is_bottom=True, use_moe_wna16_cuda=True, indtype=indtype)
        assert int(up["BLOCK_SIZE_M"]) == _W8A16_BLOCK_M
        assert int(down["BLOCK_SIZE_M"]) == _W8A16_BLOCK_M
        dm = int(down["MODE"])
        return int(up["MODE"]), _down_mode(m, dm) if tuned_shape else dm

    return _FusedState(
        w_cat=w_cat, ldy=ldy, inter=inter2 // 2, num_experts=num_experts,
        top_k=top_k, renormalize=bool(getattr(runner.router, "renormalize", True)),
        w13=w13.view(torch.uint32), w2=w2.view(torch.uint32),
        w13_scale=qcfg.w1_scale.unsqueeze(-1),
        w2_scale=qcfg.w2_scale.unsqueeze(-1),
        modes=modes)


_STATE_ATTR = "_vllm_hcu_fused_moe_state"
_UNSET = object()


def get_fused_state(block) -> _FusedState | None:
    state = getattr(block, _STATE_ATTR, _UNSET)
    if state is _UNSET:
        if torch.cuda.is_current_stream_capturing():
            return None
        state = _build_state(block)
        setattr(block, _STATE_ATTR, state)
    return state


def _moe_align(topk_ids: torch.Tensor, num_experts: int):
    """Block-align routed tokens for the BLOCK_M=16 W8A16 kernels.

    aiter's Python ``moe_align_block_size`` routes 256 experts to the SGLang
    variant (zeros + align + sort = 3 kernels); the CUB-fused moe_c kernel
    does the same in one launch.  Its only semantic difference is marking
    unused trailing blocks -1 instead of 0; the W8A16 kernels stop at
    ``num_tokens_post_pad`` so those blocks are never read.
    """
    import aiter

    numel = topk_ids.numel()
    max_padded = numel + num_experts * (_W8A16_BLOCK_M - 1)
    dev = topk_ids.device
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(triton.cdiv(max_padded, _W8A16_BLOCK_M),
                             dtype=torch.int32, device=dev)
    ntp = torch.empty(1, dtype=torch.int32, device=dev)
    aiter.moe_c_moe_align_block_size(topk_ids, num_experts, _W8A16_BLOCK_M,
                                     sorted_ids, expert_ids, ntp)
    return sorted_ids, expert_ids, ntp


class PendingMoEOutput:
    """MoE block output whose final reduction has not been materialised.

    Holding the routed partials (``cache3``), the shared-expert output and the
    shared-expert gate logits lets the consumer fold the tail into the next
    RMSNorm (see ``pending_add_rmsnorm``) instead of a separate kernel.
    """

    __slots__ = ("cache3", "shared_out", "y", "ldy", "gate_col", "top_k")

    def __init__(self, cache3, shared_out, y, ldy, gate_col, top_k):
        self.cache3 = cache3
        self.shared_out = shared_out
        self.y = y
        self.ldy = ldy
        self.gate_col = gate_col
        self.top_k = top_k


def fused_moe_block_pending(block, st: _FusedState,
                            hidden_states: torch.Tensor) -> PendingMoEOutput:
    import aiter

    x = hidden_states
    m, hidden = x.shape
    dtype = x.dtype
    top_k, inter, E, ldy = st.top_k, st.inter, st.num_experts, st.ldy

    y = torch.nn.functional.linear(x, st.w_cat)  # [m, ldy]

    act = torch.empty(m, inter, dtype=dtype, device=x.device)
    blk = min(inter, 1024)
    _strided_silu_mul_kernel[(m, inter // blk)](
        y, ldy, act, D=inter, BLOCK=blk, num_warps=4)
    shared_out, _ = block.shared_expert.down_proj(act)

    topk_w = torch.empty(m, top_k, dtype=torch.float32, device=x.device)
    topk_ids = torch.empty(m, top_k, dtype=torch.int32, device=x.device)
    # A single 64-lane wavefront: the eight successive 256-wide reductions
    # need no cross-wave sync (~4 us vs ~8 us with 4 warps).
    _strided_topk_softmax_kernel[(m,)](
        y, ldy, 2 * inter, topk_w, topk_ids, E=E, TOPK=top_k,
        RENORM=st.renormalize, num_warps=1)

    sorted_ids, expert_ids, ntp = _moe_align(topk_ids, E)
    mode_up, mode_down = st.modes(m)
    n13 = st.w13_scale.shape[1]
    cache1 = torch.empty(m, top_k, n13, dtype=dtype, device=x.device)
    aiter.moe_c_moe_gemm_marlin_w8a16(
        x, st.w13, cache1, st.w13_scale, None, sorted_ids, expert_ids, ntp,
        top_k, mode_up, 1)
    cache2 = torch.empty(m * top_k, n13 // 2, dtype=dtype, device=x.device)
    aiter.moe_c_silu_and_mul(cache2, cache1.view(-1, n13))
    cache3 = torch.empty(m, top_k, hidden, dtype=dtype, device=x.device)
    aiter.moe_c_moe_gemm_marlin_w8a16(
        cache2, st.w2, cache3, st.w2_scale, topk_w, sorted_ids, expert_ids,
        ntp, 1, mode_down, 1)
    return PendingMoEOutput(cache3, shared_out, y, ldy, 2 * inter + E, top_k)


def materialize(p: PendingMoEOutput) -> torch.Tensor:
    m, _, hidden = p.cache3.shape
    out = torch.empty(m, hidden, dtype=p.cache3.dtype, device=p.cache3.device)
    blk = 1024 if hidden % 1024 == 0 else hidden
    _moe_tail_kernel[(m, hidden // blk)](
        p.cache3, p.shared_out, p.y, p.ldy, p.gate_col, out, N=hidden,
        TOPK=p.top_k, BLOCK=blk, num_warps=4)
    return out


def pending_add_rmsnorm(p: PendingMoEOutput, residual: torch.Tensor,
                        weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma fused-add RMSNorm whose input is a pending MoE output.

    Updates ``residual`` in place (like the stock fused-add norm) and returns
    the normalised activations.
    """
    m, _, hidden = p.cache3.shape
    assert residual.is_contiguous() and residual.shape == (m, hidden)
    out = torch.empty_like(residual)
    _moe_tail_add_rmsnorm_kernel[(m,)](
        p.cache3, p.shared_out, p.y, p.ldy, p.gate_col, residual, weight,
        out, eps, N=hidden, TOPK=p.top_k, num_warps=8)
    return out


def fused_moe_block_forward(block, st: _FusedState,
                            hidden_states: torch.Tensor) -> torch.Tensor:
    return materialize(fused_moe_block_pending(block, st, hidden_states))
