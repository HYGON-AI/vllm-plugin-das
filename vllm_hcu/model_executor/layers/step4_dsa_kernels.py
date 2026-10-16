# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Step-4 DSA Triton kernels from the generated reference release.
# The migration keeps numerical function bodies unchanged. Imports are
# plugin-owned where needed, and duplicate top-level imports were removed.
# Review changes to numerical kernels against the reference parity harness.

"""Generated Step-4 minimal inference kernel.py; do not edit by hand.

Regenerate with ``python3 -m dsa_parity.build_step4_minimal_release``.
Numerical bodies come from the source manifest below.  Only merged-module
imports/metadata and the experimental Triton sliding-attention branch are
rewritten; sliding attention is the validated PyTorch SDPA implementation.
"""

from __future__ import annotations

# Source manifest (step4-minimal-inference-release-v1):
#   inference/dsa_kernels.py  sha256=7fa17fc7886bc20eaa9a2cb34c623b11ff152389442f43314199e3c8bcccd449
#   inference/dsa_meta.py  sha256=484ff7c6de71966cdc80d32ad58b091b49d730d3752f9b90b42f2e19d1df9e9f
#   inference/sparse_attn.py  sha256=22f733eb5fa0069dc11b688a2d3d3acd42125a4c379a660fef9454f783b49c2f
#   inference/dsa_attention.py  sha256=b3404cc32a8715b99bba14a0372e53e0ea5b35e582810536002dafa710494460
#   inference/fp8_gemm.py  sha256=808bad1273267b6587b582614e7745b80517d33f4a8cc79050fd2e93414ef943
#   inference/moe_kernels.py  sha256=c1c2c83d13758d9a16e6cec5795ecbd7a1081724f5d5098cda13ca8bf48a5f43
#   inference/qknorm_rope.py  sha256=f60bccb1bd62b0b6db268202ee526c501877bf66b5c6c9db5975dc6b9f5123ea


# ============================================================================
# merged source: inference/dsa_kernels.py
# ============================================================================
"""Triton kernels for Step-4's DeepSeek-style sparse attention (DSA).

Pure Triton on purpose: the reference implementations of this path depend either on
NVIDIA's CuTe-DSL package or on vLLM internals, neither of which can ship in an
open-source model release. Semantics here are pinned against the CuTe kernels at
Step-4 geometry; see ``dsa_parity/`` for the parity harness that enforces that.

Where this deviates from the CuTe kernels it is deliberate and noted at the point of
deviation. The one structural deviation: summaries are stored in a plain logical layout
rather than the CuTe cache's WGMMA XOR-swizzled byte order, because that swizzle exists
to feed Hopper WGMMA operands from shared memory and Triton chooses its own operand
layouts. Producer and consumer are both in this file, so the layout is ours to pick.
"""


import torch
import triton
import triton.language as tl

# Step-4 compresses every 8 consecutive tokens into one summary vector. The CuTe kernels
# hard-specialize on this; we keep it a parameter but never exercise anything else.
REGION_BLOCK_SIZE = 8
PROXY_DIM = 256


def _shared_memory_limit() -> int:
    """Bytes of shared memory one block may use, or 0 when that is unknowable.

    The sparse attention kernels stage a whole region-block of keys *and* values in
    shared memory, so their tile size is bounded by the device rather than by the
    geometry.  Hopper allows 227 KiB per block and the ``block_regions=16`` default is
    sized for it; a 64 KiB device (CDNA/DCU) cannot launch that tile at all and Triton
    reports OutOfResources at first launch rather than at import.  Ask the driver
    instead of hardcoding either number.
    """
    try:
        from triton.runtime.driver import driver

        return int(
            driver.active.utils.get_device_properties(torch.cuda.current_device())[
                "max_shared_mem"
            ]
        )
    except Exception:  # pragma: no cover - no driver, no device, or no such property.
        return 0


def _fit_block_regions(
    block_regions: int,
    *,
    region_size: int,
    block_d: int,
    element_size: int,
) -> int:
    """Shrink ``block_regions`` until one key+value tile fits in shared memory.

    Halved rather than reduced by one so the region count stays a power of two, which
    keeps the ``tl.dot`` operand shapes the kernels are tuned around.  Returns at least
    one region-block: a device that cannot stage even one is a real failure, and the
    OutOfResources error names the shape better than anything raised here would.
    """
    limit = _shared_memory_limit()
    if limit <= 0:
        return block_regions
    while block_regions > 1:
        if 2 * block_regions * region_size * block_d * element_size <= limit:
            break
        block_regions //= 2
    return block_regions


def shared_table_stride(block_table: torch.Tensor) -> int:
    """Row stride to pass for a block table, zero when the table is shared.

    A single-row table means "every request uses this mapping". Expressing that as a zero
    row stride rather than a branch inside the kernel keeps the two cases on one code path,
    and -- more importantly -- makes the shared case safe: the alternative, indexing a
    one-row table by a nonzero request id, reads past the tensor and returns plausible
    garbage instead of failing.
    """
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )
    return 0 if block_table.shape[0] == 1 else block_table.stride(0)


@triton.jit
def _affine_normalize(
    raw,
    mean,
    rstd,
    weight_ptr,
    bias_ptr,
    offsets,
    mask,
    weight_bias,
    HAS_BIAS: tl.constexpr,
):
    """Apply ``(x - mean) * rstd * (w + weight_bias) [+ b]`` to one dim slice.

    A device function rather than a closure because Triton cannot compile nested
    ``def`` inside a kernel, and the normalization has to be applied to three different
    dim slices of the same head: the tail, and each half of the rotated span.
    """
    scale = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (raw - mean) * rstd * (scale + weight_bias)
    if HAS_BIAS:
        result += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    return result


@triton.jit
def _store_value(value, dtype: tl.constexpr, ROUND_E4M3: tl.constexpr):
    out = value.to(dtype)
    if ROUND_E4M3:
        out = out.to(tl.float8e4nv).to(dtype)
    return out


@triton.jit
def _indexer_norm_rope_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_x_token,
    stride_out_token,
    stride_cos,
    eps,
    weight_bias,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROUND_E4M3: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Normalize one head of one token, then rotate its leading ``2 * rotary_dim`` dims.

    ``ROUND_E4M3`` additionally applies ``round_activations_e4m3`` to what is
    stored (output dtype -> e4m3 -> output dtype, the same conversions torch
    runs), so a scorer that would round the result can skip its own pass.

    ``SUBTRACT_MEAN`` selects LayerNorm over RMSNorm. The indexer needs both: q is
    RMSNorm with a zero-centered weight, k is LayerNorm with a weight and a bias. They
    differ by one reduction and are trivial to confuse, which would shift every key
    without changing any shape.

    The rotated span is written from a separate load rather than by patching the
    normalized vector in place, because Triton cannot index a value tensor by another
    tensor -- and overlapping stores from one program would depend on store ordering the
    compiler does not promise.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = token * stride_x_token + head * head_dim
    out_base = token * stride_out_token + head * head_dim

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    values = tl.load(x_ptr + base + dims, mask=dim_ok, other=0.0).to(tl.float32)

    sum_sq = tl.sum(values * values)
    if SUBTRACT_MEAN:
        # Match the deployed LayerNorm reduction and fast-rsqrt path.  Computing
        # ``mean((x - mean) ** 2)`` is algebraically equivalent, but changes the
        # FP32 reduction/rounding before every sparse-indexer key.
        mean = tl.sum(values) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = 0.0
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)

    # Dims past the rotated span keep the plain normalized value.
    tl.store(
        out_ptr + out_base + dims,
        _store_value(
            _affine_normalize(
                values,
                mean,
                rstd,
                weight_ptr,
                bias_ptr,
                dims,
                dim_ok,
                weight_bias,
                HAS_BIAS=HAS_BIAS,
            ),
            out_ptr.dtype.element_ty,
            ROUND_E4M3,
        ),
        mask=dim_ok & (dims >= 2 * rotary_dim),
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(cos_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(
        tl.float32
    )

    real = (
        _affine_normalize(
            tl.load(x_ptr + base + half, mask=half_ok, other=0.0).to(tl.float32),
            mean,
            rstd,
            weight_ptr,
            bias_ptr,
            half,
            half_ok,
            weight_bias,
            HAS_BIAS=HAS_BIAS,
        )
        .to(out_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    imaginary = (
        _affine_normalize(
            tl.load(x_ptr + base + rotary_dim + half, mask=half_ok, other=0.0).to(
                tl.float32
            ),
            mean,
            rstd,
            weight_ptr,
            bias_ptr,
            rotary_dim + half,
            half_ok,
            weight_bias,
            HAS_BIAS=HAS_BIAS,
        )
        .to(out_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    tl.store(
        out_ptr + out_base + half,
        _store_value(
            real * cos - imaginary * sin, out_ptr.dtype.element_ty, ROUND_E4M3
        ),
        mask=half_ok,
    )
    tl.store(
        out_ptr + out_base + rotary_dim + half,
        _store_value(
            real * sin + imaginary * cos, out_ptr.dtype.element_ty, ROUND_E4M3
        ),
        mask=half_ok,
    )


def _norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    weight_bias: float,
    subtract_mean: bool,
    round_e4m3: bool = False,
) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _indexer_norm_rope_kernel[(x.shape[0], num_heads)](
        x,
        out,
        weight,
        bias if bias is not None else weight,
        cos,
        sin,
        positions,
        x.stride(0),
        out.stride(0),
        cos.stride(0),
        eps,
        weight_bias,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=bias is not None,
        ROUND_E4M3=round_e4m3,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
    )
    return out


def indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    round_q: bool = False,
    contiguous_z: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the indexer's q/k/z from their fused projection.

    Args:
        index_q: ``[tokens, num_q_heads * head_dim]``.
        index_k: ``[tokens, num_k_heads * head_dim]``.
        index_z: ``[tokens, num_k_heads * head_dim]`` -- returned untouched.
        cos, sin: ``[seq, rotary_dim]``, half the width of the rotated span.
        q_norm_weight_bias: ``1.0`` for a zero-centered RMSNorm weight (``x * (1 + w)``),
            ``0.0`` otherwise. This follows the checkpoint's norm class, so it must be
            threaded through from config rather than hardcoded.
        round_q: return q already passed through ``round_activations_e4m3``.
        contiguous_z: copy a row-strided index_z; ``False`` returns it as is,
            for consumers that take its row stride.

    index_q and index_k may be row-strided (column slices of one projection);
    the normalised outputs are contiguous either way. Position values must lie
    within the supplied tables; the caller owns that scheduler invariant.
    """
    # These checks only inspect tensor metadata, without a device readback or
    # synchronization. The kernel uses packed columns and shares the cos stride
    # with sin, so incompatible layouts must fail before launching it.
    if head_dim <= 0 or num_q_heads <= 0 or num_k_heads <= 0:
        raise ValueError("head_dim and indexer head counts must be positive")
    if not 0 < 2 * rotary_dim <= head_dim:
        raise ValueError(
            f"rotated span {2 * rotary_dim} must be positive and <= head_dim {head_dim}"
        )
    if index_q.ndim != 2:
        raise ValueError("index_q must be [tokens, num_q_heads * head_dim]")
    num_tokens = index_q.shape[0]
    for name, tensor, heads in (
        ("index_q", index_q, num_q_heads),
        ("index_k", index_k, num_k_heads),
        ("index_z", index_z, num_k_heads),
    ):
        if tensor.shape != (num_tokens, heads * head_dim) or tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have matching tokens and packed head columns")
    for name, tensor in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_norm_bias", k_norm_bias),
    ):
        if tensor.shape != (head_dim,) or tensor.stride(0) != 1:
            raise ValueError(f"{name} must be a contiguous vector of length head_dim")
    if (
        cos.ndim != 2
        or cos.shape[1] != rotary_dim
        or cos.shape[0] <= 0
        or sin.shape != cos.shape
        or cos.stride(1) != 1
        or sin.stride() != cos.stride()
    ):
        raise ValueError("cos/sin must be [positions, rotary_dim] with matching row strides")
    if (
        positions.ndim != 1
        or positions.shape[0] != num_tokens
        or positions.stride(0) != 1
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("positions must be one contiguous integer per token")
    return (
        _norm_rope(
            index_q,
            q_norm_weight,
            None,
            cos,
            sin,
            positions,
            num_heads=num_q_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=q_norm_weight_bias,
            subtract_mean=False,
            round_e4m3=round_q,
        ),
        _norm_rope(
            index_k,
            k_norm_weight,
            k_norm_bias,
            cos,
            sin,
            positions,
            num_heads=num_k_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=0.0,
            subtract_mean=True,
        ),
        index_z.contiguous() if contiguous_z else index_z,
    )


@triton.jit
def _csa_compress_regions_kernel(
    index_k_ptr,
    index_z_ptr,
    token_start_ptr,
    token_count_ptr,
    summary_ptr,
    mean_ptr,
    stride_token,
    stride_head,
    stride_summary_region,
    stride_summary_head,
    num_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """One program per (region, head): compress a region's tokens into one vector.

    The weight is a softmax over the token axis of ``index_z`` taken **per dimension**,
    so this is an element-wise weighted mean, not a convex combination of whole key
    vectors:

        s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)

    ``m`` is a scalar over both axes. It cancels algebraically and only exists to keep
    ``exp`` in range -- but it is the *same* scalar the CuTe kernel uses, which is why
    the two agree in the last places rather than merely within tolerance.
    """
    region = tl.program_id(0)
    head = tl.program_id(1)

    start = tl.load(token_start_ptr + region).to(tl.int64)
    count = tl.load(token_count_ptr + region).to(tl.int64)

    tokens = tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    token_ok = tokens < count
    dim_ok = dims < proxy_dim
    mask = token_ok[:, None] & dim_ok[None, :]

    offsets = (
        (start + tokens.to(tl.int64))[:, None] * stride_token
        + head * stride_head
        + dims[None, :]
    )
    logits = tl.load(index_z_ptr + offsets, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    values = tl.load(index_k_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    shift = tl.max(tl.where(mask, logits, float("-inf")))
    weights = tl.where(mask, tl.exp(logits - shift), 0.0)
    denominator = tl.sum(weights, axis=0)
    numerator = tl.sum(weights * values, axis=0)
    summary = tl.where(
        denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
    )

    out = region * stride_summary_region + head * stride_summary_head + dims
    tl.store(summary_ptr + out, summary, mask=dim_ok)
    # Round through bfloat16 on the way to e4m3, matching the CuTe kernel. Going
    # straight from fp32 is observably different: bf16 keeps 8 mantissa bits and e4m3
    # keeps 3, so a value sitting on an e4m3 midpoint can round the other way once bf16
    # has already moved it. The parity gate compares bytes, so this is not cosmetic.
    tl.store(mean_ptr + out, summary.to(tl.bfloat16).to(tl.float8e4nv), mask=dim_ok)


def csa_region_layout(
    seq_lens: torch.Tensor,
    *,
    regions_per_seq: int,
    region_size: int = REGION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token span of every region, for sequences packed back to back.

    ``regions_per_seq`` is the stride of the region id space, not the number of regions a
    sequence actually fills -- a sequence's regions have to land at a fixed offset so the
    summary store can be indexed without a second lookup. Regions past a sequence's end
    get a count of zero and are skipped rather than written, which keeps a stale summary
    from a previous request out of the selection.
    """
    device = seq_lens.device
    lengths = seq_lens.to(torch.int64)
    token_starts = torch.cumsum(lengths, dim=0) - lengths

    regions = torch.arange(regions_per_seq, device=device, dtype=torch.int64)
    offset = regions.view(1, -1) * region_size
    count = (lengths.view(-1, 1) - offset).clamp(0, region_size)
    start = token_starts.view(-1, 1) + offset
    return start.reshape(-1).to(torch.int32), count.reshape(-1).to(torch.int32)


def csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress packed tokens into per-region summaries.

    Args:
        index_k: ``[tokens, heads, proxy_dim]`` proxy keys.
        index_z: ``[tokens, heads, proxy_dim]`` per-dimension softmax logits.
        token_start: ``[regions]`` int32, first token of each region.
        token_count: ``[regions]`` int32, tokens present in each region; 0 = skip.

    Returns:
        ``(summary_fp32, summary_fp8)``, both ``[regions, heads, proxy_dim]``. The fp8
        tensor is what selection and sparse attention consume; the fp32 one exists
        because it is the only place the unrounded value is observable, which is what
        lets a parity failure be attributed to the reduction rather than the rounding.
    """
    if index_k.shape != index_z.shape:
        raise ValueError(
            f"index_k and index_z must match, got {tuple(index_k.shape)} and "
            f"{tuple(index_z.shape)}"
        )
    if index_k.ndim != 3:
        raise ValueError(
            f"index_k must be [tokens, heads, proxy_dim], got {tuple(index_k.shape)}"
        )
    if index_k.stride(2) != 1:
        raise ValueError("index_k must be contiguous in the proxy dimension")

    regions = int(token_start.numel())
    _, heads, proxy_dim = index_k.shape
    summary = torch.zeros(
        (regions, heads, proxy_dim), device=index_k.device, dtype=torch.float32
    )
    mean = torch.zeros(
        (regions, heads, proxy_dim), device=index_k.device, dtype=torch.float8_e4m3fn
    )

    _csa_compress_regions_kernel[(regions, heads)](
        index_k,
        index_z,
        token_start,
        token_count,
        summary,
        mean,
        index_k.stride(0),
        index_k.stride(1),
        summary.stride(0),
        summary.stride(1),
        num_heads=heads,
        proxy_dim=proxy_dim,
        BLOCK_T=triton.next_power_of_2(region_size),
        BLOCK_D=triton.next_power_of_2(proxy_dim),
        # ``weights * values`` feeds a tree reduction, and a backend that contracts the
        # multiply into the summation's FMA changes the fp32 result by an ulp or two.
        # That is inside the fp32 band the gate allows, but the summary is *also* stored
        # through bfloat16 to e4m3 -- a double rounding -- and a value sitting near the
        # bf16 tie boundary then lands on a different e4m3 code. The gate compares those
        # bytes, so the reduction has to round exactly like the reference.
        **_NO_FP_FUSION,
    )
    return summary, mean


@triton.jit
def _indexer_logits_kernel(
    index_q_ptr,
    index_k_ptr,
    weights_ptr,
    out_ptr,
    seq_q,
    seq_k,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_k_token,
    stride_w_token,
    stride_w_group,
    stride_out_row,
    rows_per_group,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Weighted-ReLU indexer scores for one (query tile, key tile, provider group).

        score[q, r] = sum_h relu(dot(q_qh, k_r)) * w_qh

    The ReLU is per head and *inside* the sum, so the heads cannot be folded into a
    single matmul -- hence the loop. The per-head weights are signed, so applying
    ReLU to each dot does **not** make the final weighted sum non-negative. Exact
    zeros are still common because a score is zero whenever every head dot is
    negative; for Step-4's 4 heads per group that event is about 6.25%. Those zeros
    are what the top-k tie-break then has to resolve, so they are load-bearing,
    not noise.
    """
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    query_ok = queries < seq_q
    key_ok = keys < seq_k

    key_tile = tl.load(
        index_k_ptr + dims[:, None] + keys[None, :] * stride_k_token,
        mask=key_ok[None, :],
        other=0.0,
    )

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + queries[:, None] * stride_q_token
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + queries * stride_w_token + group * stride_w_group + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    # ``rows_per_group`` rather than ``seq_q``: when the caller writes straight into
    # a larger destination its groups are spaced by that destination's token count,
    # not by this launch's query count. It equals ``seq_q`` for the standalone form.
    rows = group * rows_per_group + queries
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def round_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the indexer's e4m3 activation rounding, then widen back to bfloat16.

    The deployed kernel quantizes indexer activations to e4m3 before the score matmul,
    even though the checkpoint stores these weights in bf16. Skipping the rounding
    changes which regions get selected, so it is part of the semantics and not an
    optimization we can decline.

    Widening back to bf16 instead of feeding e4m3 into the matmul is deliberate. It is
    lossless -- e4m3 keeps 3 mantissa bits and bf16 keeps 8 -- and it keeps the kernel
    off Triton's fp8 MMA path, which is where portability across Triton versions and
    non-NVIDIA backends breaks down. The cost is peak throughput on Hopper, not
    accuracy.
    """
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_q: int = 64,
    block_k: int = 128,
    out: torch.Tensor | None = None,
    out_row_base: int = 0,
    out_group_stride: int | None = None,
    quantize_keys: bool = True,
    q_prerounded: bool = False,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores over proxy keys.

    Args:
        index_q: ``[seq_q, groups, heads_per_group, proxy_dim]``.
        weights: ``[seq_q, groups, heads_per_group]``, already carrying the
            ``heads_per_group ** -0.5`` prescale.
        index_k: ``[seq_k, 1, proxy_dim]`` -- one shared key head (the indexer is MQA
            even though the main attention is GQA).
        out: optional destination to score into, ``[rows, cols]`` float32. Row
            ``group * out_group_stride + out_row_base + q`` gets columns
            ``[0, seq_k)``; everything else is left alone. Passing this is what lets
            a caller that already owns a full-size score buffer skip the temporary
            *and* the copy back -- see the note below on why that matters here.
        out_row_base: first destination token row, within each group's block.
        out_group_stride: rows between consecutive groups in ``out``. Defaults to
            ``seq_q``, which is the standalone packing.
        quantize_keys: apply the e4m3 activation rounding to ``index_k``. Pass
            ``False`` only when the keys are *already* e4m3-valued, in which case
            the rounding is provably the identity (every finite e4m3 value is exact
            in bf16, so widening and re-rounding returns the same byte) and doing it
            is a wasted pass over the key tile. It is not a tolerance knob: with
            keys that are not already e4m3 this changes which regions get selected.
        q_prerounded: the same argument for ``index_q``, when it came from
            ``indexer_norm_rope(round_q=True)``.

    Returns:
        ``out`` when given, else a fresh ``[groups * seq_q, seq_k]`` float32 tensor,
        group-major then query. That row order is what the region selector consumes,
        so it is a contract rather than a convenience.

    The ``out`` path exists because the natural loop around this function is
    quadratically wasteful. A caller scoring a long prefill has to bound the fp32
    score tensor -- at 65536 queries against 8192 regions that is 2 GiB -- so it
    chunks the query axis and copies each chunk back, once per group. But the query
    axis is already ``grid`` dimension 0, so the chunking buys nothing except launch
    count: measured on a 65536-token prefill it turned one launch per DSA layer into
    32, plus 2 quantizations and 4 group copies each. Writing into the caller's
    buffer removes the temporary that forced the chunking in the first place.
    """
    if index_k.shape[1] != 1:
        raise ValueError(
            f"indexer keys are MQA; expected one head, got {index_k.shape[1]}"
        )
    seq_q, groups, heads_per_group, proxy_dim = index_q.shape
    seq_k = index_k.shape[0]
    if index_k.shape[2] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} k={index_k.shape[2]}")

    quantized_q = index_q if q_prerounded else round_activations_e4m3(index_q)
    quantized_k = round_activations_e4m3(index_k) if quantize_keys else index_k

    if out is None:
        if out_row_base or out_group_stride is not None:
            raise ValueError("out_row_base/out_group_stride require an explicit out")
        out = torch.empty(
            (groups * seq_q, seq_k), device=index_q.device, dtype=torch.float32
        )
        target, rows_per_group = out, seq_q
    else:
        if out.dtype != torch.float32:
            raise ValueError(f"out must be float32, got {out.dtype}")
        if out.shape[1] < seq_k:
            raise ValueError(f"out has {out.shape[1]} columns, need {seq_k}")
        rows_per_group = seq_q if out_group_stride is None else out_group_stride
        needed = out_row_base + (groups - 1) * rows_per_group + seq_q
        if needed > out.shape[0]:
            raise ValueError(f"out has {out.shape[0]} rows, need {needed}")
        # Shifting the base pointer instead of passing ``out_row_base`` as an
        # argument: every group's rows move down by the same amount, so the
        # kernel's ``group * rows_per_group + q`` lands correctly on the view, and
        # the specialization key gains no int arg that varies per request.
        target = out[out_row_base:] if out_row_base else out

    _indexer_logits_kernel[
        (triton.cdiv(seq_q, block_q), triton.cdiv(seq_k, block_k), groups)
    ](
        quantized_q,
        quantized_k,
        weights,
        target,
        seq_q,
        seq_k,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        quantized_k.stride(0),
        weights.stride(0),
        weights.stride(1),
        target.stride(0),
        rows_per_group,
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )
    return out


# Prefill selection packs one entry as ``phys_region | (valid_tokens << 24)``.
# These are ``tl.constexpr`` because Triton kernels cannot read plain module globals.
PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
RADIX_BITS = tl.constexpr(8)
RADIX_BUCKETS = tl.constexpr(256)
RADIX_PASSES = tl.constexpr(4)


@triton.jit
def _ordered_float_key(scores):
    """Map fp32 values to monotonically ordered unsigned-32 keys held in int64.

    Negative IEEE-754 values reverse their bit ordering, while non-negative values
    become ordered after setting the sign bit. Keeping the resulting 32-bit pattern
    in a non-negative int64 lets ordinary Triton integer comparisons implement the
    required unsigned ordering. Branching on ``scores < 0`` deliberately maps
    ``-0.0`` and ``+0.0`` to the same key, matching numeric equality and CuTe.

    NaNs follow CuTe's raw ordered-float behavior and sort above ``+inf``. Production
    indexer scores are expected to be finite; this helper does not add a costly
    full-tensor finiteness scan to the hot path.
    """
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(
        scores < 0.0,
        (~bits) & 0xFFFFFFFF,
        bits | 0x80000000,
    )


@triton.jit
def _radix_topk_threshold(
    logits_ptr,
    row,
    stride_logits_row,
    visible,
    budget,
    seq_regions,
    BLOCK_R: tl.constexpr,
):
    """Find the exact score threshold separating the top ``budget`` entries of one row.

    Returns ``(threshold, remaining)``: the fp32 bit pattern of the ``budget``-th largest
    score, and how many entries *at* that exact key still belong in the selection. A
    threshold plus a tie count is what lets the caller take a **prefix** of a tied group
    rather than all or none of it, without ever materializing a sort.

    Scores may be signed. :func:`_ordered_float_key` converts their IEEE-754 bit
    patterns into unsigned-monotone keys before the radix passes.
    """
    threshold = budget.to(tl.int64) * 0
    remaining = budget
    bucket_ids = tl.arange(0, RADIX_BUCKETS)

    # Narrow the threshold one byte at a time, most significant first.
    for radix_pass in tl.static_range(RADIX_PASSES):
        shift = 32 - RADIX_BITS * (radix_pass + 1)
        histogram = tl.zeros((RADIX_BUCKETS,), dtype=tl.int32)
        for base in tl.range(0, seq_regions, BLOCK_R):
            regions = base + tl.arange(0, BLOCK_R)
            live = regions < visible
            scores = tl.load(
                logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
            )
            keys = _ordered_float_key(scores)
            if radix_pass == 0:
                # No high bits are fixed yet. Testing them would need a shift by 32,
                # which is undefined.
                matches = live
            else:
                high = shift + RADIX_BITS
                matches = live & ((keys >> high) == (threshold >> high))
            # Park non-matching lanes in bucket 0 and subtract them back out. Relying on
            # tl.histogram to drop an out-of-range sentinel would corrupt the bucket
            # totals if it clamps instead -- and the totals are what pick the cut, so the
            # failure would look like a wrong threshold rather than a bad histogram.
            buckets = tl.where(
                matches,
                ((keys >> shift) & (RADIX_BUCKETS - 1)).to(tl.int32),
                0,
            )
            counts = tl.histogram(buckets, RADIX_BUCKETS)
            parked = tl.sum((~matches).to(tl.int32))
            histogram += counts - tl.where(bucket_ids == 0, parked, 0)

        total = tl.sum(histogram)
        above = total - tl.cumsum(histogram, axis=0)
        # ``above`` is non-increasing, so the buckets satisfying above < remaining form a
        # suffix; the smallest of them is the one holding the cut.
        chosen = tl.min(tl.where(above < remaining, bucket_ids, RADIX_BUCKETS))
        remaining -= tl.sum(tl.where(bucket_ids > chosen, histogram, 0))
        # Cast before shifting the MSB byte. An int32 bucket >= 128 would otherwise
        # become negative first and sign-extend into the int64 threshold.
        threshold |= chosen.to(tl.int64) << shift

    return threshold, remaining


@triton.jit
def _region_topk_pack_kernel(
    logits_ptr,
    lengths_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    logical_out_ptr,
    seq_regions,
    stride_logits_row,
    stride_table_row,
    stride_out_row,
    stride_logical_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    WRITE_LOGICAL_IDS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions of one row and pack them.

    Selection is ``torch.topk`` with ties broken by ascending region id: order the
    visible scores by ``(-score, id)`` and keep the first ``min(topk, visible)``. Ties at
    the cut take a *prefix* of the tied group, they do not drop it -- and that case is
    common rather than exotic, because post-ReLU indexer scores are exactly zero about
    6.25% of the time, so the group sitting at the cut is usually the zero group.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if WRITE_LOGICAL_IDS:
        tl.store(
            logical_out_ptr + row * stride_logical_row + slots,
            -1,
            mask=slots < topk,
        )
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr,
        row,
        stride_logits_row,
        visible,
        budget,
        seq_regions,
        BLOCK_R=BLOCK_R,
    )

    # Emit in ascending region order. The CuTe selector emits in radix-bucket order, but
    # that order is not part of the contract -- consumers compare sets -- and ascending
    # is both cheaper here and what the standalone metadata converter produces.
    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        safe_regions = tl.where(selected, regions, 0)
        pages = tl.load(
            block_table_ptr
            + request * stride_table_row
            + safe_regions // regions_per_page,
            mask=selected,
            other=0,
        )
        physical = pages * regions_per_page + safe_regions % regions_per_page
        valid_tokens = tl.minimum(
            tl.maximum(q_position + 1 - safe_regions * region_size, 0), region_size
        )
        packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
        # An unmapped page is a legal block-table state, not an error. It has to become a
        # ``-1`` slot rather than an address computed from a negative page, which would
        # read from before the cache.
        packed = tl.where(pages >= 0, packed, -1)
        tl.store(
            out_ptr + row * stride_out_row + positions,
            packed,
            mask=selected & (positions < topk),
        )
        if WRITE_LOGICAL_IDS:
            tl.store(
                logical_out_ptr + row * stride_logical_row + positions,
                regions,
                mask=selected & (positions < topk),
            )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


@triton.jit
def _region_topk_ids_kernel(
    logits_ptr,
    lengths_ptr,
    out_ptr,
    seq_regions,
    stride_logits_row,
    stride_out_row,
    topk: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions and emit **logical ids**, ascending.

    The same selection as :func:`_region_topk_pack_kernel` with the packing left off.
    Prefill and decode pack the identical selection into two different words, and the
    decode word needs the logical region id, which the prefill word does not carry -- so
    decode metadata cannot be derived from prefill metadata, only from the raw ids.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr,
        row,
        stride_logits_row,
        visible,
        budget,
        seq_regions,
        BLOCK_R=BLOCK_R,
    )

    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(
            out_ptr + row * stride_out_row + positions,
            regions,
            mask=selected & (positions < topk),
        )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


@triton.jit
def _region_topk_ids_resident_kernel(
    logits_ptr,
    lengths_ptr,
    out_ptr,
    seq_regions,
    stride_logits_row,
    stride_out_row,
    topk: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """:func:`_region_topk_ids_kernel` for a row that fits in one tile.

    Same selection and output, bit for bit. The row is loaded once and kept in registers
    as signed-ordered int32 keys (the :func:`_ordered_float_key` order shifted by
    ``2**31``), and the threshold is found by bisecting the key range on
    ``count(keys >= mid)``. Each step is one block reduction, against four histogram
    passes plus a tiled emit in the radix kernel; the search also stops as soon as some
    ``mid`` leaves exactly ``budget`` keys at or above it. The emit is a single cumsum
    with the greater and tied counts packed into the two 16-bit halves of an int32, so
    rows are limited to ``< 2**15`` regions. 100 -> 52us at 64 rows x 8704 regions.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    regions = tl.arange(0, BLOCK_R)
    live = regions < visible
    scores = tl.load(
        logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
    )
    bits = scores.to(tl.int32, bitcast=True)
    keys = tl.where(scores < 0.0, bits ^ 0x7FFFFFFF, bits & 0x7FFFFFFF)
    keys = tl.where(live, keys, -2147483648)

    # Invariant: count(keys >= lo) >= budget > count(keys >= hi). The threshold is the
    # largest such lo, which is itself a key.
    lo = tl.min(tl.where(live, keys, 2147483647)).to(tl.int64)
    hi = tl.max(keys).to(tl.int64) + 1
    done = budget * 0
    while (hi - lo > 1) & (done == 0):
        mid = lo + (hi - lo) // 2
        above = keys >= mid.to(tl.int32)
        count = tl.sum(above.to(tl.int32))
        if count == budget:
            lo = tl.min(tl.where(above, keys, 2147483647)).to(tl.int64)
            done = 1
        elif count > budget:
            lo = mid
        else:
            hi = mid
    threshold = lo.to(tl.int32)

    # Ascending region order, ties at the cut taken as a prefix -- as the tiled emit.
    greater = live & (keys > threshold)
    tied = live & (keys == threshold)
    prefix = tl.cumsum(greater.to(tl.int32) | (tied.to(tl.int32) << 16), axis=0)
    greater_incl = prefix & 0xFFFF
    tied_incl = prefix >> 16
    remaining = budget - tl.max(greater_incl)
    selected = greater | (tied & (tied_incl <= remaining))
    positions = greater_incl + tl.minimum(tied_incl, remaining) - 1
    tl.store(
        out_ptr + row * stride_out_row + positions,
        regions,
        mask=selected & (positions < topk),
    )


# Widest row the register-resident selector takes; wider rows use the tiled radix one.
_RESIDENT_TOPK_MAX_REGIONS = 16384


def region_topk_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int,
    block_r: int = 2048,
) -> torch.Tensor:
    """Top-k region selection, returning logical region ids in ascending order.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed.
        lengths: ``[rows]`` int32, candidate region count per row. For DSA this is the
            **history** length, ``q_position // region_size``, which excludes the region
            containing the query itself -- that one is appended by the metadata builder
            rather than made to compete.

    Returns:
        ``[rows, topk]`` int32 logical region ids, ``-1`` padding the tail.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")

    rows, seq_regions = logits.shape
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    if seq_regions <= _RESIDENT_TOPK_MAX_REGIONS:
        _region_topk_ids_resident_kernel[(rows,)](
            logits,
            lengths,
            out,
            seq_regions,
            logits.stride(0),
            out.stride(0),
            topk=int(topk),
            # HCU: >= 2 for the same reason as below.
            BLOCK_R=max(2, triton.next_power_of_2(seq_regions)),
            BLOCK_TOPK=triton.next_power_of_2(int(topk)),
            num_warps=8,
        )
        return out
    _region_topk_ids_kernel[(rows,)](
        logits,
        lengths,
        out,
        seq_regions,
        logits.stride(0),
        out.stride(0),
        topk=int(topk),
        # HCU: BLOCK_R must stay >= 2. The DCU Triton backend fails to lower a
        # 1-wide region tile (the scalar/vector compare at the packing site comes
        # out with a mismatched register class and clang rejects the module), and
        # next_power_of_2(1) == 1 does occur in practice: seq_regions is 1 for any
        # sequence still inside its first region, which is every warmup step.
        # Widening the tile is behaviour-neutral -- the loop masks with
        # ``regions < visible`` -- so the extra lane is inert.
        BLOCK_R=max(2, min(block_r, triton.next_power_of_2(seq_regions))),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
        # One program per row and few rows at decode: wider programs finish the
        # serial radix passes sooner (117 -> 104us at 64 rows x 8.3k regions).
        num_warps=8,
    )
    return out


def region_topk_pack(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    block_r: int = 2048,
    _logical_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Top-k region selection fused with prefill metadata packing.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed, where
            ``rows = groups * seq_q`` in group-major order.
        lengths: ``[rows]`` int32, causally visible region count per row.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page. A table with a single row is
            treated as shared by every request, regardless of ``request_indices``.
        request_indices: ``[rows]`` int32, which block-table row each score row belongs
            to. Defaults to all-zero.

    Returns:
        ``[rows, topk]`` int32, each valid slot holding
        ``phys_region | (valid_tokens << 24)``, with ``-1`` padding the tail. A row fills
        exactly ``min(topk, visible)`` slots.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )

    rows, seq_regions = logits.shape
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    _region_topk_pack_kernel[(rows,)](
        logits,
        lengths,
        q_positions,
        block_table,
        request_indices,
        out,
        out if _logical_out is None else _logical_out,
        seq_regions,
        logits.stride(0),
        shared_table_stride(block_table),
        out.stride(0),
        out.stride(0) if _logical_out is None else _logical_out.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        WRITE_LOGICAL_IDS=_logical_out is not None,
        # HCU: see region_topk_ids -- a 1-wide region tile does not compile on DCU.
        BLOCK_R=max(2, min(block_r, triton.next_power_of_2(seq_regions))),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
    )
    return (out, _logical_out) if _logical_out is not None else out


# Preserve the first definition so later merged modules cannot silently
# overwrite a different metadata encoding.
_MERGED_PREFILL_REGION_VALID_SHIFT = PREFILL_REGION_VALID_SHIFT


# ============================================================================
# merged source: inference/dsa_meta.py
# ============================================================================
"""Stage 4: turn a region selection into the sparse-attention metadata.

This is the layer between the selector and the attention kernels, and it carries the two
facts about DSA that are easiest to get wrong because nothing crashes when you do.

**The query's own region never competes.** The selector's candidate range is
``[0, q_position // region_size)`` -- strictly *past* regions. The region containing the
query itself is appended afterwards, unconditionally. So a query always attends to at
least itself, and ``region_counts`` is one more than the number of history slots. Letting
the current region into the top-k instead would work for most rows and silently drop the
query's own context on the rows where it lost, which is exactly the kind of degradation
that shows up as slightly worse output rather than as a bug.

**There is no sliding window here.** ``sliding_window: 512`` in the checkpoint config
belongs to the sliding-attention layers, which are a different set of layers. The upstream
metadata converter does take a ``window`` argument -- and every DSA call site passes
``window=0``. When it is non-zero it *replaces* the single current-region append with the
full run of regions covering ``[q_position - window + 1, q_position]``, unioned with the
history; that generalisation is deliberately not implemented here, because implementing an
unused mode would leave it ungated.

The prefill and decode words are different, and so are the conventions around them:

============  ==============================  ================================
\\             prefill                         decode
============  ==============================  ================================
dtype         int32                           int64
word          ``phys | valid_tokens << 24``    ``start_token | phys << 32``
valid_tokens  packed                          derived from ``kv_seqlen``
dropped slot  left in place as ``-1``         compacted out
tail padding  ``-1``                          **zero**
order         ascending logical region        ascending packed word (physical)
============  ==============================  ================================

The zero padding on the decode side is the sharp edge: a zero word is a structurally valid
slot pointing at physical region 0 with a full complement of tokens. Nothing but
``region_counts`` distinguishes it from a real selection.
"""


import triton
import triton.language as tl


PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
DECODE_PHYS_SHIFT = tl.constexpr(32)


@triton.jit
def _append_current_region_kernel(
    history_ptr,
    history_ids_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    ids_out_ptr,
    counts_ptr,
    stride_history_row,
    stride_history_ids_row,
    stride_table_row,
    stride_out_row,
    stride_ids_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    WRITE_LOGICAL_IDS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Widen prefill history metadata to ``topk + 1`` slots and append the current region.

    The history is copied verbatim -- including its ``-1`` slots, which stay *in place*
    rather than being compacted, because prefill's ``region_counts`` is positional
    (``min(current_region, topk) + 1``) and not a count of live entries. The appended slot
    lands at exactly the index the selector stopped filling.

    The two stores are kept **disjoint** (``slots != history_count``) rather than letting
    the second overwrite the first. Writing the whole row and then patching one slot is a
    cross-thread race: the row store is a vector store spread over lanes, the patch is a
    scalar store from one lane, and nothing orders them. It loses roughly one row in a
    thousand, so it survives a small gate and fails a large one. The upstream kernel
    carries a comment about this exact hazard, which suggests it was found the same way.
    """
    row = tl.program_id(0)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)
    current_region = tl.maximum(q_position // region_size, 0)
    history_count = tl.minimum(current_region, topk)

    slots = tl.arange(0, BLOCK_K)
    history = tl.load(
        history_ptr + row * stride_history_row + slots, mask=slots < topk, other=-1
    )
    tl.store(
        out_ptr + row * stride_out_row + slots,
        history,
        mask=(slots < topk + 1) & (slots != history_count),
    )

    if WRITE_LOGICAL_IDS:
        history_ids = tl.load(
            history_ids_ptr + row * stride_history_ids_row + slots,
            mask=slots < topk,
            other=-1,
        )
        tl.store(
            ids_out_ptr + row * stride_ids_out_row + slots,
            history_ids,
            mask=(slots < topk + 1) & (slots != history_count),
        )

    page = tl.load(
        block_table_ptr
        + request * stride_table_row
        + current_region // regions_per_page
    )
    physical = page * regions_per_page + current_region % regions_per_page
    valid_tokens = tl.minimum(
        tl.maximum(q_position + 1 - current_region * region_size, 0), region_size
    )
    packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
    tl.store(
        out_ptr + row * stride_out_row + history_count,
        tl.where(page >= 0, packed, -1),
    )
    if WRITE_LOGICAL_IDS:
        tl.store(
            ids_out_ptr + row * stride_ids_out_row + history_count,
            tl.where(page >= 0, current_region, -1),
        )
    tl.store(counts_ptr + row, history_count + 1)


@triton.jit
def _decode_pack_kernel(
    ids_ptr,
    seqlens_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    counts_ptr,
    stride_ids_row,
    stride_table_row,
    stride_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Pack selected logical regions into the int64 decode word.

    Surviving slots are compacted with a cumsum scatter, keeping the selector's ascending
    logical order. The upstream converter sorts into physical-major order instead; the two
    coincide whenever the block table is monotonic, and otherwise differ only in the
    attention's accumulation order. The sort cost 14.4 us vs 5.8 us here (64 rows, topk
    512). The current region is then written at the surviving count, so it stays last,
    which is also where the reference puts it.
    """
    row = tl.program_id(0)
    request = tl.load(request_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int32)
    valid_regions = (kv_seqlen + region_size - 1) // region_size

    slots = tl.arange(0, BLOCK_K)
    ids = tl.load(ids_ptr + row * stride_ids_row + slots, mask=slots < topk, other=-1)

    live = (slots < topk) & (ids >= 0) & (ids < valid_regions)
    safe_ids = tl.where(live, ids, 0)
    pages = tl.load(
        block_table_ptr + request * stride_table_row + safe_ids // regions_per_page,
        mask=live,
        other=-1,
    )
    live = live & (pages >= 0)

    physical = pages * regions_per_page + safe_ids % regions_per_page
    packed = (safe_ids * region_size).to(tl.int64) | (
        physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    live_i32 = live.to(tl.int32)
    position = tl.cumsum(live_i32, axis=0) - 1
    count = tl.sum(live_i32)
    # Selected words land in [0, count), zeros in (count, topk]; slot ``count`` is left to
    # the scalar store below. The three slot sets are disjoint, so there is no
    # cross-thread race between the vector stores and the scalar one -- see
    # :func:`_append_current_region_kernel`.
    tl.store(out_ptr + row * stride_out_row + position, packed, mask=live)
    tl.store(
        out_ptr + row * stride_out_row + slots,
        tl.zeros_like(packed),
        mask=(slots > count) & (slots < topk + 1),
    )

    current_region = tl.maximum((kv_seqlen - 1) // region_size, 0)
    current_page = tl.load(
        block_table_ptr
        + request * stride_table_row
        + current_region // regions_per_page
    )
    current_live = (current_region < valid_regions) & (current_page >= 0)
    current_physical = (
        current_page * regions_per_page + current_region % regions_per_page
    )
    current_packed = (current_region * region_size).to(tl.int64) | (
        current_physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    # Written unconditionally so the slot is defined even when the current region is
    # unmapped, in which case it becomes ordinary zero padding.
    tl.store(
        out_ptr + row * stride_out_row + count,
        tl.where(current_live, current_packed, 0),
    )
    tl.store(counts_ptr + row, count + current_live.to(tl.int32))


def prefill_sparse_meta(
    logits: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    return_logical_ids: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Select regions and build prefill sparse metadata.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed,
            group-major (``rows = num_kv_groups * total_q``). The scores are weighted
            sums of per-head ReLU terms and unconstrained signed indexer weights.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page, which yields a ``-1`` slot rather
            than an error.
        request_indices: ``[rows]`` int32, which block-table row each score row uses.
            Defaults to all-zero, i.e. every row shares one table.

    Returns:
        ``(packed [rows, topk + 1] int32, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_prefill`. When
        ``return_logical_ids`` is true, a third ``[rows, topk + 1]`` int32 tensor is
        returned; it contains the logical region id for each live slot and ``-1`` padding.

    The history length is derived here as ``q_position // region_size`` rather than taken
    from the caller, because getting it wrong by one is the difference between the current
    region being force-included and it competing for a slot.
    """
    rows = int(logits.shape[0])
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    history_lengths = torch.div(
        q_positions.to(torch.int32), region_size, rounding_mode="floor"
    ).contiguous()

    if return_logical_ids:
        history, logical_history = region_topk_pack(
            logits,
            history_lengths,
            q_positions,
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
            _logical_out=torch.empty(
                (rows, int(topk)), device=logits.device, dtype=torch.int32
            ),
        )
    else:
        history = region_topk_pack(
            logits,
            history_lengths,
            q_positions,
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
        )
        logical_history = None

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int32)
    logical_ids = (
        torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int32)
        if return_logical_ids
        else None
    )
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _append_current_region_kernel[(rows,)](
        history,
        history if logical_history is None else logical_history,
        q_positions,
        block_table,
        request_indices,
        packed,
        packed if logical_ids is None else logical_ids,
        counts,
        history.stride(0),
        history.stride(0) if logical_history is None else logical_history.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        packed.stride(0) if logical_ids is None else logical_ids.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        WRITE_LOGICAL_IDS=return_logical_ids,
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    if return_logical_ids:
        return packed, counts, logical_ids
    return packed, counts


def decode_rows_meta(
    kv_seqlens: torch.Tensor,
    rows: int,
    request_indices: torch.Tensor | None = None,
    region_size: int = REGION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``decode_sparse_meta``'s per-row ``(request_indices, history_lengths)``."""
    num_reqs = int(kv_seqlens.shape[0])
    if rows % num_reqs:
        raise ValueError(f"{rows} score rows do not divide into {num_reqs} requests")

    if request_indices is None:
        request_indices = (
            torch.arange(num_reqs, device=kv_seqlens.device, dtype=torch.int32)
            .repeat(rows // num_reqs)
            .contiguous()
        )
    # Decode's query position is the last token, so the history is everything strictly
    # before the current region.
    history_lengths = torch.div(
        (kv_seqlens.to(torch.int32) - 1).clamp_min(0),
        region_size,
        rounding_mode="floor",
    )[request_indices.to(torch.long)].contiguous()
    return request_indices, history_lengths


def decode_sparse_meta(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    rows_meta: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build decode sparse metadata.

    Args:
        logits: ``[num_kv_groups * num_reqs, seq_regions]`` float32 indexer scores,
            which may be signed, group-major. The selector orders them numerically rather
            than by the magnitude encoded in their IEEE-754 bit pattern.
        kv_seqlens: ``[num_reqs]`` int32, context length *including* the current token.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page.
        request_indices: ``[rows]`` int32. Defaults to the group-major layout implied by
            the row count, ``arange(num_reqs)`` tiled once per KV group.
        rows_meta: ``decode_rows_meta``'s result for these lengths and this row
            count, replacing ``request_indices``. It depends on neither the scores
            nor the layer, so a caller scoring many layers builds it once.

    Returns:
        ``(packed [rows, topk + 1] int64, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_decode`.

    ``region_counts`` here is a count of *surviving* entries, so it varies with how much of
    the block table is mapped. The tail beyond it is zero, and a zero word is not
    self-identifying -- the count is the only bound.
    """
    rows, _ = logits.shape
    if rows_meta is None:
        rows_meta = decode_rows_meta(kv_seqlens, rows, request_indices, region_size)
    request_indices, history_lengths = rows_meta

    ids = region_topk_ids(logits, history_lengths, topk=topk)

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int64)
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _decode_pack_kernel[(rows,)](
        ids,
        kv_seqlens,
        block_table,
        request_indices,
        packed,
        counts,
        ids.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    return packed, counts


assert (
    PREFILL_REGION_VALID_SHIFT.value == _MERGED_PREFILL_REGION_VALID_SHIFT.value == 24
), "merged DSA modules disagree on PREFILL_REGION_VALID_SHIFT"
del _MERGED_PREFILL_REGION_VALID_SHIFT
_MERGED_DECODE_PHYS_SHIFT = DECODE_PHYS_SHIFT


# ============================================================================
# merged source: inference/sparse_attn.py
# ============================================================================
"""Triton sparse attention over the selected regions -- the DSA forward itself.

Separate from ``dsa_kernels`` because this is the only stage that touches the main KV
cache and the main 192-dim heads; everything there operates on the 256-dim indexer proxy.

Semantics are pinned against ``token_wise_flash_attn_prefill_sm90_gqa_func`` and its
reference in ``optimus_jit/tests/test_sparse_gqa_steptron_fwd_reference.py``. Three
details of that contract are worth stating up front because none is guessable:

* Each query row carries its **own** region list. Queries cannot share a K tile the way
  dense attention does, so the natural unit of work is one (query, KV group) pair rather
  than a query tile.
* There is **no causal mask** in the attention itself. Causality is already baked into the
  packed metadata: ``valid_tokens`` says how much of each region the query may see, and
  the selector only ever offered causally visible regions.
* Softmax runs over the gathered token set only, and an empty selection yields a zero
  output with ``lse = -inf`` rather than a NaN.
"""


import triton
import triton.language as tl

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)
DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Attention for one query against one KV group's selected regions.

    Online softmax over region tiles. The accumulator is per (head, dim) and the running
    max is per head, exactly as in dense flash attention -- the only difference is that
    the K/V rows come from a gather driven by the packed region list instead of a
    contiguous range.

    The cache may hold ``float8_e4m3`` rather than the query's dtype. Both dequantization
    scales stay out of the fp8 operands: ``k_scale`` is pre-folded into ``softmax_scale``
    by the wrapper, so it lands on the fp32 score tile, and ``v_scale`` multiplies the
    fp32 accumulator at the end. Scaling the operands instead would re-quantize them,
    which is the same rule the fp8 expert GEMM follows.
    """
    query = tl.program_id(0)
    group = tl.program_id(1)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    for base in tl.range(0, topk, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < count
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = meta >> REGION_VALID_SHIFT
        # A slot is live only if it was within the count, not a -1 sentinel, and actually
        # exposes tokens. The third condition is not redundant: a region whose first token
        # is past the query position packs a valid id with zero visible tokens.
        live = slot_ok & (meta >= 0) & (valid > 0)

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )
        safe_tokens = tl.where(token_ok, tokens, 0)

        keys = tl.load(
            k_ptr
            + safe_tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        # Widen to the compute dtype before either dot. This is what keeps the later
        # ``probabilities.to(values.dtype)`` honest: against an fp8 cache that cast would
        # otherwise quantize the softmax probabilities to e4m3.
        keys = keys.to(queries.dtype)
        values = values.to(queries.dtype)

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(
            token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY
        )

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        # A tile can be entirely masked -- every slot exhausted, or a row with no
        # selection at all. Then both maxima are -inf and ``m - m`` is NaN, which would
        # poison the accumulator for the rest of the loop. Substituting a finite shift
        # keeps the arithmetic well defined and leaves the running state untouched.
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = (
        tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0) * v_scale
    )
    tl.store(
        out_ptr
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr + query * stride_lse_token + group * heads_per_group + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_prefill_tiled_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    list_ptr,
    logical_list_ptr,
    lengths_ptr,
    mask_ptr,
    q_start_ptr,
    q_count_ptr,
    q_positions_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_list_tile,
    stride_list_slot,
    stride_logical_tile,
    stride_logical_slot,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    num_tiles,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Query-tiled sparse prefill over a per-tile union of selected regions.

    The selector produces one region list per query, but neighbouring queries usually
    select nearly the same regions. This kernel loads each union region once per query
    tile and uses a bitset to decide which query owns each token. Tiles are never allowed
    to cross a request boundary by :func:`_build_prefill_tiles`, so a logical region id
    maps to one physical cache region within a tile.

    ``v_scale`` and the ``k_scale`` already folded into ``softmax_scale`` carry an fp8
    cache's dequantization; see :func:`_sparse_attn_prefill_kernel` for why they are
    applied to the fp32 side rather than to the operands.
    """
    tile = tl.program_id(0)
    head = tl.program_id(1)

    q_local = tl.arange(0, BLOCK_Q)
    # Tile metadata is group-major; Q/output token indices are group-local.
    metadata_start = tl.load(q_start_ptr + tile).to(tl.int32)
    group = metadata_start // total_q
    q_start = metadata_start - group * total_q
    q_count = tl.load(q_count_ptr + tile).to(tl.int32)
    q_index = q_start + q_local
    q_ok = (q_local < q_count) & (q_index < total_q)

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim

    queries = tl.load(
        q_ptr
        + q_index[:, None] * stride_q_token
        + (group * heads_per_group + head) * stride_q_head
        + dims[None, :],
        mask=q_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )

    running_max = tl.full((BLOCK_Q,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    positions = tl.load(
        q_positions_ptr + metadata_start + q_local,
        mask=q_ok,
        other=-1,
    ).to(tl.int32)
    words_of_q = q_local // 32
    bits_of_q = q_local % 32
    offsets_in_region = tl.arange(0, region_size)
    union_count = tl.load(lengths_ptr + tile).to(tl.int32)

    for base in tl.range(0, union_count, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < union_count
        physical = tl.load(
            list_ptr + tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr + tile * stride_logical_tile + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + tile * stride_mask_tile
            + words_of_q[:, None] * stride_mask_word
            + slots[None, :] * stride_mask_slot,
            mask=q_ok[:, None] & slot_ok[None, :],
            other=0,
        )
        selected = ((words >> bits_of_q[:, None]) & 1) == 1

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_pos = tl.reshape(
            logical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_live = tl.reshape(
            tl.broadcast_to(slot_ok[:, None], (BLOCK_R, region_size)),
            (BLOCK_R * region_size,),
        )
        causal = token_pos[None, :] <= positions[:, None]
        keep = causal & token_live[None, :] & q_ok[:, None]
        keep = keep & tl.reshape(
            tl.broadcast_to(selected[:, :, None], (BLOCK_Q, BLOCK_R, region_size)),
            (BLOCK_Q, BLOCK_R * region_size),
        )

        safe_tokens = tl.where(token_live, tokens, 0)
        keys = tl.load(
            k_ptr
            + safe_tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_live[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_live[:, None] & dim_ok[None, :],
            other=0.0,
        )
        keys = keys.to(queries.dtype)
        values = values.to(queries.dtype)

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))
        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected_any = running_sum > 0
    output = (
        tl.where(selected_any[:, None], accumulator / running_sum[:, None], 0.0)
        * v_scale
    )
    tl.store(
        out_ptr
        + q_index[:, None] * stride_out_token
        + (group * heads_per_group + head) * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None] & dim_ok[None, :],
    )
    tl.store(
        lse_ptr + q_index * stride_lse_token + group * heads_per_group + head,
        tl.where(selected_any, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=q_ok,
    )


@triton.jit
def _sparse_attn_prefill_tiled_hd192_kernel(
    q_ptr,
    q_start_ptr,
    q_count_ptr,
    q_positions_ptr,
    k_ptr,
    v_ptr,
    list_ptr,
    logical_list_ptr,
    lengths_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_list_tile,
    stride_list_slot,
    stride_logical_tile,
    stride_logical_slot,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    scale_log2,
    v_scale,
    q_scale,
    total_q,
    num_tiles,
    heads_per_group: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DK: tl.constexpr,
    QK_FP8: tl.constexpr,
    PV_FP8: tl.constexpr,
    P_SCALE: tl.constexpr,
) -> None:
    """``_sparse_attn_prefill_tiled_kernel`` specialized to ``head_dim = 3 * DK``.

    Same contract, same metadata, same results within a bf16 ULP. Four changes
    took this from 196.0 ms to 5.1 ms at 65K on one HCU (38.4x), measured 2026-09-22:

    * **The head dim is chunked, not padded.** The general kernel rounds
      ``head_dim`` up to a power-of-two ``BLOCK_D``, so 192 becomes 256 and a
      quarter of both dots is masked-off waste. ``192 = 3 * 64`` exactly, so three
      chunks have no waste at all and every MFMA lane does useful work.
    * **The softmax scale is folded into an ``exp2`` argument.** The general kernel
      multiplies the whole ``[BLOCK_Q, BLOCK_R*region_size]`` score tile by
      ``softmax_scale`` and calls ``exp``; folding it into a base-2 exponent makes
      that multiply disappear. It was the single largest instruction in the loop.
    * **``num_stages=1``.** The wrapper passes it. The AMD backend's pipeliner is a
      4x pessimization here and its extra buffering is what pushes ``BLOCK_R=16``
      past the shared-memory limit.

    Two more are load-side and came later, from a per-kernel trace against the
    vendor's own tiled prefill (which ran 1.19x faster on an otherwise identical
    207-call workload). Only one of the two survived measurement:

    * **K is loaded already transposed**, dropping ``tl.trans`` from the QK dots --
      the same argument as in the decode hd192 kernel below. Bit-identical, and
      worth 7.52 ms -> 7.40 ms (1.6%) at 8192 queries / topk 512 / ``BLOCK_R`` 8 on
      a bf16 cache. On an *fp8* cache the same change is worth 13.92 ms -> 9.17 ms
      (1.52x): transposing a tile of fp8 bytes costs far more than transposing bf16,
      so the 1.6% measured on bf16 badly understates it.
    * **Hoisting the V gathers above the softmax does not help and is not done.**
      It reads like free overlap, and the vendor kernel does it, but measured on
      the same shape it costs 7.40 ms -> 7.56 ms: the loads are already in flight
      by then and issuing them earlier only lengthens their live ranges.

    Neither accounts for the 1.19x. The traced runs differed in cache dtype too, and
    that is where it was: with everything else held fixed this kernel ran 7.41 ms on
    a bf16 cache and 13.92 ms on an fp8 one, a 1.9x penalty for reading fp8 bytes
    through a bf16 dot. The K-transpose above took that to 1.26x on its own.
    ``QK_FP8``/``PV_FP8`` close the rest, mirroring the decode hd192 kernel below:

    * ``QK_FP8`` feeds both QK operands in as fp8 so the dot reaches the hardware fp8
      MMA (K=32 per instruction against bf16's K=16, and no software byte-unpack).
      Widening K to the query dtype first -- what the ``.to()`` does when this is
      false -- costs about 17 integer ops per cache element *and* still runs the
      half-rate bf16 MMA. Q is quantized once at kernel entry; ``q_scale`` is folded
      back out of ``scale_log2`` by the caller, so it never reaches the output.
    * ``PV_FP8`` does the same for the second dot, and is the one that touches P.
      e4m3 has three mantissa bits and post-max-subtraction probabilities sit near
      the bottom of its exponent range, so P is prescaled by ``P_SCALE`` first; that
      factor divides out of the normalizer and never reaches the output.

    ``PV_FP8`` without ``QK_FP8`` is the worst of both -- K still gets widened *and*
    P loses precision -- so the caller refuses that combination.

    ``QK_FP8`` *alone* is legal but is not an operating point: measured on the same
    shape it is the slowest of the four, worse than not using fp8 math at all. At
    production tile size (``block_regions=16`` -> BLOCK_R=16, 128 tokens/tile):

        bf16 cache  37.79 ms   256 VGPRs, 50 spills
        fp8 widen   36.86 ms   256 VGPRs,  6 spills   (what this kernel did before)
        fp8 qk      12.40 ms   256 VGPRs, 91 spills
        fp8 full     6.23 ms   253 VGPRs,  0 spills

    The spill column is the whole story. Under ``QK_FP8`` alone the loop body has to
    hold an fp8 K tile for the first dot *and* a bf16-widened V tile for the second,
    so both operand layouts are live at once and it blows through the 256-VGPR
    budget. ``PV_FP8`` is what removes bf16 from the loop entirely, and only then
    does the body fit. So for prefill the two flags are effectively one switch:
    both, or neither. Note ``fp8 full`` also beats the bf16 cache outright (6.1x) --
    the narrower loads more than pay for the quantization once the MMA is fp8.

    **Tile-size crossover:** this speedup only holds at the production tile size.
    At ``block_regions=4`` (BLOCK_R=4, 32 tokens/tile) the speedup inverts:

        bf16 cache  17.41 ms   (smallest at small tile)
        fp8 widen   37.89 ms
        fp8 qk      36.29 ms
        fp8 full    33.54 ms

    Small tiles cannot amortize the quantization overhead, so bf16 wins. vLLM's call
    site hardcoded ``block_regions=4``; moving it to 16 unblocks the 6x win.

    What that costs in accuracy, against the bf16-cache run on the same inputs (at
    ``block_regions=16``): ``fp8 widen`` is already at 1.4e-02 max-abs / 0.99930 mean
    cosine purely from the stored cache, and ``fp8 full`` moves that to 1.9e-02 /
    0.99866. The extra error is from quantizing Q and P, which the fp32 reference
    does not model at all, which is why stage 5 rejects these flags by default the
    way stage 6 does.

    ``scale_log2`` is ``softmax_scale * log2(e)`` with any fp8 ``k_scale`` already
    folded in, exactly as the general kernel's ``softmax_scale`` is; ``v_scale``
    still multiplies the fp32 output. The lse is stored in natural log to keep the
    documented contract, so the ``log2`` is converted on the way out.
    """
    tile = tl.program_id(0)
    head = tl.program_id(1)

    q_local = tl.arange(0, BLOCK_Q)
    # Tile metadata is group-major; Q/output token indices are group-local.
    metadata_start = tl.load(q_start_ptr + tile).to(tl.int32)
    group = metadata_start // total_q
    q_start = metadata_start - group * total_q
    q_count = tl.load(q_count_ptr + tile).to(tl.int32)
    q_index = q_start + q_local
    q_ok = (q_local < q_count) & (q_index < total_q)

    dk = tl.arange(0, DK)
    q_head_off = (group * heads_per_group + head) * stride_q_head
    qr0 = tl.load(
        q_ptr + q_index[:, None] * stride_q_token + q_head_off + (0 * DK + dk)[None, :],
        mask=q_ok[:, None],
        other=0.0,
    )
    qr1 = tl.load(
        q_ptr + q_index[:, None] * stride_q_token + q_head_off + (1 * DK + dk)[None, :],
        mask=q_ok[:, None],
        other=0.0,
    )
    qr2 = tl.load(
        q_ptr + q_index[:, None] * stride_q_token + q_head_off + (2 * DK + dk)[None, :],
        mask=q_ok[:, None],
        other=0.0,
    )
    # ``qr*`` stay in the query dtype and remain the dtype reference for everything
    # downstream; ``q*`` is what enters the dot, quantized once here rather than per
    # region tile. q_scale is divided back out of scale_log2 by the caller.
    if QK_FP8:
        q0 = (qr0.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q1 = (qr1.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q2 = (qr2.to(tl.float32) * q_scale).to(tl.float8e4nv)
    else:
        q0 = qr0
        q1 = qr1
        q2 = qr2
    running_max = tl.full((BLOCK_Q,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_Q, DK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_Q, DK), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_Q, DK), dtype=tl.float32)

    positions = tl.load(
        q_positions_ptr + metadata_start + q_local, mask=q_ok, other=-1
    ).to(tl.int32)
    words_of_q = q_local // 32
    bits_of_q = q_local % 32
    offsets_in_region = tl.arange(0, region_size)
    union_count = tl.load(lengths_ptr + tile).to(tl.int32)
    k_group_off = group * stride_k_group
    v_group_off = group * stride_v_group

    for base in tl.range(0, union_count, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < union_count
        physical = tl.load(
            list_ptr + tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr + tile * stride_logical_tile + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + tile * stride_mask_tile
            + words_of_q[:, None] * stride_mask_word
            + slots[None, :] * stride_mask_slot,
            mask=q_ok[:, None] & slot_ok[None, :],
            other=0,
        )
        selected = ((words >> bits_of_q[:, None]) & 1) == 1

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_pos = tl.reshape(
            logical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_live = tl.reshape(
            tl.broadcast_to(slot_ok[:, None], (BLOCK_R, region_size)),
            (BLOCK_R * region_size,),
        )
        causal = token_pos[None, :] <= positions[:, None]
        keep = causal & token_live[None, :] & q_ok[:, None]
        keep = keep & tl.reshape(
            tl.broadcast_to(selected[:, :, None], (BLOCK_Q, BLOCK_R, region_size)),
            (BLOCK_Q, BLOCK_R * region_size),
        )

        safe_tokens = tl.where(token_live, tokens, 0)
        # K is addressed as ``[DK, tokens]`` instead of being loaded ``[tokens, DK]``
        # and run through ``tl.trans``. The transpose is a full LDS round trip per
        # tile and costs more than the dot it feeds; reading the same bytes in the
        # order the MFMA wants is bit-identical. The decode hd192 kernel below
        # already loads K this way.
        #
        # Under QK_FP8 the tile goes into the dot as fp8; otherwise it is cast to the
        # query's compute dtype, and an fp8 cache is dequantized by that cast. Either
        # way the descale itself rides on scale_log2 (the caller folded k_scale in and
        # q_scale out) and on v_scale at the end.
        k_base = k_ptr + safe_tokens[None, :] * stride_k_token + k_group_off
        k0 = tl.load(
            k_base + (0 * DK + dk)[:, None], mask=token_live[None, :], other=0.0
        )
        k1 = tl.load(
            k_base + (1 * DK + dk)[:, None], mask=token_live[None, :], other=0.0
        )
        k2 = tl.load(
            k_base + (2 * DK + dk)[:, None], mask=token_live[None, :], other=0.0
        )
        if not QK_FP8:
            k0 = k0.to(qr0.dtype)
            k1 = k1.to(qr0.dtype)
            k2 = k2.to(qr0.dtype)
        # QK accumulated across the three head-dim chunks; exactly 192 wide, so
        # unlike the BLOCK_D=256 path there are no masked lanes to pay for.
        scores = tl.dot(q0, k0, out_dtype=tl.float32)
        scores = tl.dot(q1, k1, acc=scores, out_dtype=tl.float32)
        scores = tl.dot(q2, k2, acc=scores, out_dtype=tl.float32)
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)

        # The scale rides into exp2 rather than multiplying the score tile.
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1) * scale_log2)
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp2(running_max - shift))
        probabilities = tl.exp2(scores * scale_log2 - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)

        # V is gathered after the softmax. Hoisting these three loads above it to
        # overlap the gather latency with the transcendentals is the obvious move and
        # was measured: 7.40 ms -> 7.56 ms at 8192 queries / topk 512. The loads are
        # already in flight by then and the earlier issue only lengthens live ranges.
        v_base = v_ptr + safe_tokens[:, None] * stride_v_token + v_group_off
        v0 = tl.load(
            v_base + (0 * DK + dk)[None, :], mask=token_live[:, None], other=0.0
        )
        v1 = tl.load(
            v_base + (1 * DK + dk)[None, :], mask=token_live[:, None], other=0.0
        )
        v2 = tl.load(
            v_base + (2 * DK + dk)[None, :], mask=token_live[:, None], other=0.0
        )
        if PV_FP8:
            weights = (probabilities * P_SCALE).to(tl.float8e4nv)
        else:
            weights = probabilities.to(qr0.dtype)
            v0 = v0.to(qr0.dtype)
            v1 = v1.to(qr0.dtype)
            v2 = v2.to(qr0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(weights, v0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(weights, v1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(weights, v2, out_dtype=tl.float32)
        running_max = tile_max

    selected_any = running_sum > 0
    # running_sum accumulates the unscaled probabilities while acc* accumulated
    # ``P * P_SCALE``, so the normalizer carries P_SCALE back out here and neither it
    # nor q_scale reaches the output. running_sum is also what the lse is built from
    # below, which is why it is the one that stays unscaled.
    if PV_FP8:
        inv = tl.where(selected_any, 1.0 / (running_sum * P_SCALE), 0.0) * v_scale
    else:
        inv = tl.where(selected_any, 1.0 / running_sum, 0.0) * v_scale
    out_head_off = (group * heads_per_group + head) * stride_out_head
    tl.store(
        out_ptr
        + q_index[:, None] * stride_out_token
        + out_head_off
        + (0 * DK + dk)[None, :],
        (acc0 * inv[:, None]).to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_ptr
        + q_index[:, None] * stride_out_token
        + out_head_off
        + (1 * DK + dk)[None, :],
        (acc1 * inv[:, None]).to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_ptr
        + q_index[:, None] * stride_out_token
        + out_head_off
        + (2 * DK + dk)[None, :],
        (acc2 * inv[:, None]).to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    # running_max/running_sum are base-2; the stored lse is natural log, as the
    # general kernel's is, so convert rather than changing the contract.
    tl.store(
        lse_ptr + q_index * stride_lse_token + group * heads_per_group + head,
        tl.where(
            selected_any,
            (running_max + tl.log2(running_sum)) * 0.6931471805599453,
            NEGATIVE_INFINITY,
        ),
        mask=q_ok,
    )


# Row count above which tile boundaries are found on device. The device path costs a fixed
# ~0.37 ms (one .item() sync plus a handful of small launches) and the host path costs about
# 56 ns per row, so they cross between 4096 and 8192 rows; measured, the device path is
# 20x faster at 32k tokens and ~3x slower on a 1k-row batch.
_TILE_BOUNDS_DEVICE_MIN_ROWS = 8192


def _build_tile_bounds(
    request_indices: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    block_q: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut prefill rows into tiles at request boundaries, then into ``block_q`` pieces.

    Returns ``(q_starts, q_counts)`` int32, tiles in increasing row order. A tile never
    spans two requests: a logical region id only means something within one request.

    Two implementations with identical output; see ``_TILE_BOUNDS_DEVICE_MIN_ROWS`` for why
    the small-batch case keeps the host loop.
    """
    device = request_indices.device
    if rows < _TILE_BOUNDS_DEVICE_MIN_ROWS:
        # vLLM lays prefill rows out contiguously by request, so runs are found by a single
        # forward scan of this one small vector; all large metadata stays on device.
        request_cpu = request_indices.to(device="cpu", dtype=torch.int64).tolist()
        q_starts: list[int] = []
        q_counts: list[int] = []
        for group in range(num_kv_groups):
            group_start = group * total_q
            group_end = group_start + total_q
            run_start = group_start
            while run_start < group_end:
                request = request_cpu[run_start]
                run_end = run_start + 1
                while run_end < group_end and request_cpu[run_end] == request:
                    run_end += 1
                for start in range(run_start, run_end, block_q):
                    q_starts.append(start)
                    q_counts.append(min(block_q, run_end - start))
                run_start = run_end
        return (
            torch.tensor(q_starts, device=device, dtype=torch.int32),
            torch.tensor(q_counts, device=device, dtype=torch.int32),
        )

    ids = request_indices.to(torch.int64)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    starts_run = torch.ones(rows, device=device, dtype=torch.bool)
    starts_run[1:] = ids[1:] != ids[:-1]
    # A group boundary always cuts, even when the request id repeats across it -- groups
    # occupy contiguous row ranges and each carries its own copy of every request.
    starts_run |= (row_ids % total_q) == 0

    run_starts = starts_run.nonzero(as_tuple=True)[0]
    run_ends = torch.empty_like(run_starts)
    run_ends[:-1] = run_starts[1:]
    run_ends[-1] = rows
    run_lengths = run_ends - run_starts

    # Splitting a run of length L into block_q pieces is arithmetic, not iteration.
    tiles_per_run = -(-run_lengths // block_q)
    num_tiles = int(tiles_per_run.sum().item())
    run_of_tile = torch.repeat_interleave(
        torch.arange(run_starts.numel(), device=device, dtype=torch.int64),
        tiles_per_run,
    )
    first_tile_of_run = tiles_per_run.cumsum(0) - tiles_per_run
    index_in_run = (
        torch.arange(num_tiles, device=device, dtype=torch.int64)
        - first_tile_of_run[run_of_tile]
    )

    offset = index_in_run * block_q
    q_starts_d = run_starts[run_of_tile] + offset
    q_counts_d = torch.clamp(run_lengths[run_of_tile] - offset, max=block_q)
    return q_starts_d.to(torch.int32), q_counts_d.to(torch.int32)


# Peak size of the dense per-query bitmap staged by :func:`_build_tile_masks`. Chosen to
# match the footprint of the int64 accumulator the atomic implementation used to allocate
# at the recorded 172k-token prefill budget, so this is not a new memory high-water mark.
_TILE_MASK_DENSE_BUDGET_BYTES = 84 * 2**20


def _build_tile_masks(
    tile_of: torch.Tensor,
    q_local: torch.Tensor,
    entry_slots: torch.Tensor,
    *,
    num_tiles: int,
    words: int,
    union: int,
    block_q: int,
    device: torch.device,
    budget: int = _TILE_MASK_DENSE_BUDGET_BYTES,
) -> torch.Tensor:
    """Pack per-query region membership into ``[num_tiles, words, union]`` int32 bitmaps.

    Each ``(tile, query, slot)`` triple is written at most once -- ``q_local`` identifies
    the query within its tile and every live slot maps to a distinct logical region -- so
    no two entries ever land on the same bit. The obvious spelling,
    ``index_put_(accumulate=True)`` into an int64 accumulator, therefore does a bitwise OR
    with a carry that provably never happens, and pays for ~29M atomic adds at 16k tokens
    to do it: measured 23 ms of a 40 ms tile build, and 49 ms of 79 ms at 32k.

    Instead, scatter into a dense ``[tile, query, slot]`` bool with plain stores and fold
    the 32 queries of each output word into their bit weights with a coalesced reduction.
    That is ~10x faster, but a dense bitmap over every tile at once would reach 336 MB at
    the recorded prefill budget, 4x the accumulator it replaces. Slabbing over tiles caps
    it at ``budget`` and keeps ~8.5x, because the live entries of a tile range stay one
    contiguous slice -- so a slab costs a ``searchsorted`` lookup, not a scan of all
    entries. Slabbing over *words* instead would cost one full rescan per word and give
    back half the win.
    """
    out = torch.empty((num_tiles, words, union), device=device, dtype=torch.int32)
    if num_tiles == 0 or tile_of.numel() == 0:
        out.zero_()
        return out

    # searchsorted needs tile_of sorted. It is: it comes from repeat_interleave over row
    # order, boolean-masked, and both preserve order. Asserting keeps a future reordering
    # of the caller from silently producing wrong masks instead of failing.
    if not bool((tile_of[1:] >= tile_of[:-1]).all()):
        raise ValueError("tile_of must be non-decreasing to slab by tile range")

    weights = 1 << torch.arange(32, device=device, dtype=torch.int64)
    per_tile = max(1, block_q * union)
    slab = max(1, min(num_tiles, budget // per_tile))

    edges = torch.arange(0, num_tiles + slab, slab, device=device, dtype=tile_of.dtype)
    bounds = torch.searchsorted(tile_of, edges).tolist()

    staging = torch.empty((slab, block_q, union), device=device, dtype=torch.bool)
    for index, base in enumerate(range(0, num_tiles, slab)):
        end = min(base + slab, num_tiles)
        low, high = bounds[index], bounds[index + 1]
        view = staging[: end - base]
        view.zero_()
        view[tile_of[low:high] - base, q_local[low:high], entry_slots[low:high]] = True
        for word in range(words):
            queries = view[:, word * 32 : (word + 1) * 32, :]
            accumulated = (queries.to(torch.int64) * weights[None, :, None]).sum(dim=1)
            # Reinterpret the low 32 bits as signed: bit 31 set means a negative int32.
            out[base:end, word, :] = (
                accumulated - (accumulated >= 2**31).to(torch.int64) * 2**32
            ).to(torch.int32)
    return out


def _build_prefill_tiles(
    logical_regions: torch.Tensor,
    packed_regions: torch.Tensor,
    q_positions: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    region_size: int,
    block_q: int,
) -> tuple[torch.Tensor, ...]:
    """Build physical union lists and per-query membership masks for tiled prefill.

    A tile is cut at every request boundary before it is split into ``block_q`` queries.
    That invariant is important: a logical region number is meaningful only within one
    request, while the physical cache region is request-specific. The returned tensors are
    all on the input device and can be passed directly to the Triton kernel.
    """
    rows, slots = logical_regions.shape
    expected_rows = num_kv_groups * total_q
    if packed_regions.shape != logical_regions.shape:
        raise ValueError(
            "logical and packed prefill metadata must have the same shape, got "
            f"{tuple(logical_regions.shape)} and {tuple(packed_regions.shape)}"
        )
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} metadata rows, got {rows}")
    if q_positions.numel() == total_q:
        q_positions = q_positions.repeat(num_kv_groups)
    if q_positions.numel() != rows:
        raise ValueError(f"expected {rows} query positions, got {q_positions.numel()}")
    if request_indices.numel() != rows:
        raise ValueError(
            f"expected {rows} request indices, got {request_indices.numel()}"
        )
    if block_q <= 0 or block_q & (block_q - 1):
        raise ValueError(f"block_q must be a positive power of two, got {block_q}")

    device = logical_regions.device
    if rows == 0:
        raise ValueError("cannot build prefill tiles for an empty query batch")
    q_starts_d, q_counts_d = _build_tile_bounds(
        request_indices,
        total_q=total_q,
        num_kv_groups=num_kv_groups,
        block_q=block_q,
        rows=rows,
    )
    num_tiles = int(q_starts_d.numel())
    if num_tiles == 0:
        raise ValueError("cannot build prefill tiles for an empty query batch")
    tile_ids = torch.arange(num_tiles, device=device, dtype=torch.int64)
    tile_of_row = torch.repeat_interleave(tile_ids, q_counts_d.to(torch.int64))
    if tile_of_row.numel() != rows:
        raise ValueError("request boundaries do not cover all prefill rows")

    live = (logical_regions >= 0) & (packed_regions >= 0)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    # One index computation for all four gathers. Spelling these as ``x[live]`` recomputes
    # the mask's nonzero positions once per gather, and for the two row-length vectors it
    # also gathers through an ``expand``ed rows*slots view instead of reading the small
    # tensor. Taking the flat positions once and mapping them back with ``// slots`` costs
    # a divide and saves ~2.4x on this block (11 ms at 32k tokens).
    flat_live = live.reshape(-1).nonzero(as_tuple=True)[0]
    row_of_entry = flat_live // slots
    tile_of = tile_of_row.index_select(0, row_of_entry)
    q_local = (row_ids - q_starts_d[tile_of_row].to(torch.int64)).index_select(
        0, row_of_entry
    )
    logical = logical_regions.reshape(-1).index_select(0, flat_live).to(torch.int64)
    physical = (
        packed_regions.reshape(-1).index_select(0, flat_live).to(torch.int64)
        & ((1 << 24) - 1)
    ).to(torch.int32)

    num_regions = int(logical.max().item()) + 1 if logical.numel() else 1
    present = torch.zeros((num_tiles, num_regions), device=device, dtype=torch.bool)
    present[tile_of, logical] = True
    lengths = present.sum(dim=1).to(torch.int32)
    # Rounded up to a power of two so the Triton specialization key is stable. ``union``
    # reaches the kernel only as a stride (tile_phys.stride(0), mask.stride(0) and
    # mask.stride(1)), and Triton keys those int args on whether they divide 16 -- so a
    # union of 80 and a union of 81 compile separately, at ~1.4 s each, for identical
    # code. Rounding pins every union >= 16 into the divisible-by-16 class, which is one
    # class instead of two, and does it identically on all 8 ranks so they stop skewing
    # apart and blocking each other in the per-DSA-layer allreduce.
    #
    # Safe on three counts. ``lengths`` is untouched, and the kernel bounds its region
    # loop by ``lengths[tile]`` (``slot_ok = slots < union_count``), so the padded tail is
    # never read; the three tensors below are zero-allocated anyway. And BLOCK_R comes out
    # of ``min(block_regions, next_power_of_2(union))``, which next_power_of_2 leaves
    # unchanged -- shared-memory footprint is exactly what it was.
    union = triton.next_power_of_2(max(1, int(lengths.max().item())))

    # Mapping through logical ids is safe because a tile never crosses a request. It also
    # keeps the temporary bitmap proportional to sequence length rather than to the
    # allocator's (potentially very sparse) physical block ids.
    physical_by_logical = torch.zeros(
        (num_tiles, num_regions), device=device, dtype=torch.int32
    )
    physical_by_logical[tile_of, logical] = physical
    tile, logical_slot = present.nonzero(as_tuple=True)
    slot_of = present.cumsum(dim=1).to(torch.int64) - 1
    tile_phys = torch.zeros((num_tiles, union), device=device, dtype=torch.int32)
    tile_logical = torch.zeros((num_tiles, union), device=device, dtype=torch.int32)
    tile_phys[tile, slot_of[tile, logical_slot]] = physical_by_logical[
        tile, logical_slot
    ]
    tile_logical[tile, slot_of[tile, logical_slot]] = logical_slot.to(torch.int32)

    words = (block_q + 31) // 32
    entry_slots = slot_of[tile_of, logical]
    mask = _build_tile_masks(
        tile_of,
        q_local,
        entry_slots,
        num_tiles=num_tiles,
        words=words,
        union=union,
        block_q=block_q,
        device=device,
    )
    return q_starts_d, q_counts_d, tile_phys, tile_logical, lengths, mask


def sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    block_regions: int = 16,
    logical_regions: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    request_indices: torch.Tensor | None = None,
    block_q: int = 128,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    q_scale: float = 1.0,
    qk_fp8: bool = False,
    pv_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over per-query selected regions.

    Args:
        query: ``[total_q, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, i.e. the paged
            cache already flattened across pages. Flattening is safe because a region's
            tokens are contiguous in the physical token space by construction. May hold
            ``float8_e4m3`` instead of the query dtype; see ``k_scale``/``v_scale``.
        packed_regions: ``[num_kv_groups * total_q, topk]`` int32 from the selector,
            each slot ``phys_region | (valid_tokens << 24)``, ``-1`` padding the tail.
        region_counts: ``[num_kv_groups * total_q]`` int32, how many slots to read. This
            truncates *positionally* -- it does not re-select by score.
        logical_regions: optional logical ids parallel to ``packed_regions``. When supplied,
            the query-tiled implementation is used and ``q_positions`` is required.
        q_positions: absolute positions parallel to the prefill query rows (or one vector
            of length ``total_q`` shared by all KV groups).
        request_indices: request id parallel to metadata rows; tiles are split at changes
            in this vector so per-request logical ids never alias.
        block_q: query tile width for the tiled implementation; must be a power of two.
        k_scale, v_scale: per-tensor dequantization scales for an fp8 cache, i.e. the
            true tensor is ``cache * scale``. Both default to 1.0, which is also what the
            deployed sliding-attention path uses -- its fp8 kernels are compiled with
            ``FP8_NO_SCALES`` and ignore descales entirely, so a DSA layer quantizing at
            some other scale would put the two attention families on different footings.
        q_scale: prescale applied to the query before it is quantized to fp8, when
            ``qk_fp8`` is on. It only has to keep the query inside e4m3's range (2**-9
            subnormal to 448); Step-4 queries sit near 1 after the norm, so the default
            1.0 is right. Divided back out of ``softmax_scale``, so like
            ``k_scale``/``v_scale`` it never reaches the output.
        qk_fp8: on the ``head_dim=192`` tiled path with a ``float8_e4m3fn`` key cache,
            quantize the query to e4m3 and feed both operands of the QK dot in as fp8,
            reaching the hardware fp8 MMA instead of unpacking every cache byte into a
            half-rate bf16 dot. **Off by default, and deliberately not implied by an fp8
            cache.** The fp8 cache alone is still a bf16-grade operator, because the
            kernel and the reference read the same fp8 bytes and the storage error
            cancels; quantizing the query is new error that no fp32 reference models, and
            it lands outside stage 5's bf16 envelope (see ``operator_parity/README.md``).
            Turning it on is an accuracy decision that needs an end-to-end score, not a
            kernel decision. Same contract as the decode path's flag of the same name.
        pv_fp8: additionally quantize the probabilities and keep V in fp8 for the second
            dot. Requires ``qk_fp8``. The accuracy cost of the whole fp8 arithmetic path
            is concentrated here, since it is the one that touches P. Same caveat, more
            of it.

    Returns:
        ``(out, lse)`` shaped ``[total_q, num_heads, head_dim]`` and
        ``[total_q, num_heads]``. Rows with no selected region get a zero output and
        ``lse = -inf``.

    The caller must guarantee the selected regions are distinct. Our selector does (the
    stage 3 gate checks it), but a union-with-sliding-window path would not, and duplicate
    regions would silently double-weight those tokens in the softmax rather than fail.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * total_q:
        raise ValueError(
            f"expected {num_kv_groups * total_q} metadata rows, got {packed_regions.shape[0]}"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {tuple(key_cache.shape)} vs {tuple(value_cache.shape)}"
        )

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    # k_scale rides here rather than on the loaded keys: the product lands on the fp32
    # score tile, where it costs nothing and cannot re-quantize an fp8 operand.
    scale = scale * k_scale

    # Only OCP e4m3 goes into the dot as fp8. The AMD-native ``float8_e4m3fnuz`` is
    # rejected by the Triton backend here, and reinterpreting its bytes as e4m3fn would
    # be off by an exponent bias, so refuse rather than quietly produce a plausible
    # wrong answer. Checked before the path split so the untiled and general kernels
    # reject the flags too rather than silently ignoring them.
    if qk_fp8 and key_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"qk_fp8 needs a float8_e4m3fn key cache, got {key_cache.dtype}"
        )
    if pv_fp8:
        if not qk_fp8:
            raise ValueError("pv_fp8 requires qk_fp8; alone it is slower and lossier")
        if value_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"pv_fp8 needs a float8_e4m3fn value cache, got {value_cache.dtype}"
            )
    if (qk_fp8 or pv_fp8) and not (
        head_dim == 192
        and query.dtype in (torch.bfloat16, torch.float16)
        and logical_regions is not None
    ):
        raise ValueError(
            "qk_fp8/pv_fp8 are implemented only on the head_dim=192 tiled prefill path"
        )

    out = torch.empty_like(query)
    lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    # The staging budget is set by the *compute* dtype, not the cache dtype: the kernels
    # widen K/V to the query dtype right after the load, so an fp8 cache stages exactly as
    # many bytes as a bf16 one. Passing the cache's element size would double the fitted
    # BLOCK_R against fp8 and hit OutOfResources at first launch -- the very failure this
    # function exists to prevent.
    compute_element_size = query.element_size()

    if logical_regions is not None:
        if q_positions is None:
            raise ValueError("q_positions is required for tiled sparse prefill")
        if request_indices is None:
            request_indices = torch.zeros(
                (num_kv_groups * total_q,), device=query.device, dtype=torch.int32
            )
        q_positions = q_positions.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        request_indices = request_indices.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        if q_positions.numel() == total_q:
            q_positions = q_positions.repeat(num_kv_groups)
        if logical_regions.shape != packed_regions.shape:
            raise ValueError(
                "logical_regions and packed_regions must have identical shapes"
            )
        q_starts, q_counts, tile_phys, tile_logical, lengths, mask = (
            _build_prefill_tiles(
                logical_regions.contiguous(),
                packed_regions.contiguous(),
                q_positions,
                request_indices,
                total_q=total_q,
                num_kv_groups=num_kv_groups,
                region_size=region_size,
                block_q=block_q,
            )
        )
        num_tiles = int(q_starts.numel())
        union = int(tile_phys.shape[1])
        # head_dim 192 gets the chunked/exp2 specialization: 6.1x at 65K, same
        # results within a bf16 ULP. Anything else keeps the general BLOCK_D path,
        # which is correct for every head_dim but pads non-powers-of-two.
        if head_dim == 192 and query.dtype in (torch.bfloat16, torch.float16):
            _sparse_attn_prefill_tiled_hd192_kernel[
                (num_tiles, heads_per_group)
            ](
                query,
                q_starts,
                q_counts,
                q_positions,
                key_cache,
                value_cache,
                tile_phys,
                tile_logical,
                lengths,
                mask,
                out,
                lse,
                query.stride(0),
                query.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                value_cache.stride(0),
                value_cache.stride(1),
                tile_phys.stride(0),
                tile_phys.stride(1),
                tile_logical.stride(0),
                tile_logical.stride(1),
                mask.stride(0),
                mask.stride(1),
                mask.stride(2),
                out.stride(0),
                out.stride(1),
                lse.stride(0),
                (scale / q_scale if qk_fp8 else scale) * 1.4426950408889634,
                v_scale,
                q_scale,
                total_q,
                num_tiles,
                heads_per_group=heads_per_group,
                region_size=region_size,
                BLOCK_Q=block_q,
                BLOCK_R=_fit_block_regions(
                    min(block_regions, triton.next_power_of_2(union)),
                    region_size=region_size,
                    block_d=192,
                    element_size=compute_element_size,
                ),
                DK=64,
                QK_FP8=qk_fp8,
                PV_FP8=pv_fp8,
                # e4m3 bottoms out at 2**-9 subnormal and post-max-subtraction
                # probabilities sit below that, so P is lifted before the cast. Same
                # value and same reasoning as the decode path.
                P_SCALE=256.0,
                num_warps=4,
                num_stages=1,
            )
            return out, lse
        _sparse_attn_prefill_tiled_kernel[(num_tiles, heads_per_group)](
            query,
            key_cache,
            value_cache,
            tile_phys,
            tile_logical,
            lengths,
            mask,
            q_starts,
            q_counts,
            q_positions,
            out,
            lse,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            tile_phys.stride(0),
            tile_phys.stride(1),
            tile_logical.stride(0),
            tile_logical.stride(1),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            scale,
            v_scale,
            total_q,
            num_tiles,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
            region_size=region_size,
            BLOCK_Q=block_q,
            BLOCK_D=block_d,
            BLOCK_R=_fit_block_regions(
                min(block_regions, triton.next_power_of_2(union)),
                region_size=region_size,
                block_d=block_d,
                element_size=compute_element_size,
            ),
            num_warps=4,
            num_stages=2,
        )
        return out, lse

    _sparse_attn_prefill_kernel[(total_q, num_kv_groups)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        scale,
        v_scale,
        total_q,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_D=block_d,
        BLOCK_R=_fit_block_regions(
            min(block_regions, triton.next_power_of_2(topk)),
            region_size=region_size,
            block_d=block_d,
            element_size=compute_element_size,
        ),
    )
    return out, lse


@triton.jit
def _sparse_attn_decode_hd192_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    q_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DK: tl.constexpr,
    QK_FP8: tl.constexpr,
    PV_FP8: tl.constexpr,
    P_SCALE: tl.constexpr,
) -> None:
    """Decode kernel specialized for head_dim=192 with head_dim chunking.

    Same semantics as the general decode kernel, but head_dim is hardcoded to
    3 * DK to avoid padding waste.

    ``softmax_scale`` arrives base-2 here (the caller folded in log2(e)) and the online
    softmax runs on exp2, as the hd192 *prefill* kernel already does. The stored lse is
    still natural log -- converted on the way out rather than changing the contract --
    so the merge function is unaffected.

    Three further details are performance, not semantics, and two of them are load-side:

    * K is indexed as ``[DK, tokens]`` rather than loaded as ``[tokens, DK]`` and
      run through ``tl.trans``. The transpose is a full LDS round trip per tile and
      costs more than the dot it feeds; addressing the same bytes in the transposed
      order is bit-identical and roughly 1.7x faster.
    * ``QK_FP8`` keeps an fp8 cache in fp8 all the way into ``tl.dot``, which is the
      only way to reach the hardware fp8 MMA (K=32 per instruction against bf16's
      K=16, and no software byte-unpack). Widening K to the query dtype first --
      which is what the ``.to()`` below does when ``QK_FP8`` is false -- costs about
      17 integer ops per cache element *and* still runs the half-rate bf16 MMA.
      Q is quantized once at kernel entry to match; ``q_scale`` is folded back out
      of ``softmax_scale`` by the caller, so it never reaches the output.
    * ``PV_FP8`` does the same for the second dot. It is the expensive one
      accuracy-wise and the only one that touches P: e4m3 has three mantissa bits and
      post-max-subtraction probabilities sit near the bottom of its exponent range,
      so P is prescaled by ``P_SCALE`` first. That factor divides out of the
      normalizer below and never reaches the output.

    ``PV_FP8`` without ``QK_FP8`` is the worst of both -- K still gets widened *and*
    P loses precision -- so the caller refuses that combination.
    """
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dk = tl.arange(0, DK)
    head_ok = heads < heads_per_group

    qh = (group * heads_per_group + heads) * stride_q_head
    qr0 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (0 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr1 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (1 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr2 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (2 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    if QK_FP8:
        # 3 * BLOCK_H * DK elements quantized once, against the whole selected cache
        # that now never gets widened.
        q0 = (qr0.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q1 = (qr1.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q2 = (qr2.to(tl.float32) * q_scale).to(tl.float8e4nv)
    else:
        q0, q1, q2 = qr0, qr1, qr2

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    kg = group * stride_k_group
    vg = group * stride_v_group

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        # ``slots < window_end`` is not redundant: ``slots_per_split`` is a ceiling
        # division, so it is a multiple of BLOCK_R only by luck. The production
        # width 513 over 16 splits gives 33 slots per split against BLOCK_R=16, and
        # the last tile of every split would otherwise reach 15 slots into the next
        # split's window. Both splits then accumulate those regions and the merge
        # double-counts them.
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        # Load K already transposed as ``[DK, tokens]``. On the fp8 path the tile goes
        # into the dot as fp8; otherwise it is cast to the query's compute dtype, and an
        # fp8 cache is dequantized by that cast. Either way the descale itself rides on
        # softmax_scale (which the caller has already multiplied by k_scale, and divided
        # by q_scale) and on v_scale at the end.
        kb = k_ptr + tokens[None, :] * stride_k_token + kg
        k0 = tl.load(kb + (0 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k1 = tl.load(kb + (1 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k2 = tl.load(kb + (2 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        if not QK_FP8:
            k0 = k0.to(qr0.dtype)
            k1 = k1.to(qr0.dtype)
            k2 = k2.to(qr0.dtype)

        s = tl.dot(q0, k0, out_dtype=tl.float32)
        s = tl.dot(q1, k1, acc=s, out_dtype=tl.float32)
        s = tl.dot(q2, k2, acc=s, out_dtype=tl.float32)
        s = s * softmax_scale
        s = tl.where(token_ok[None, :] & head_ok[:, None], s, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(s, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp2(running_max - shift))

        probabilities = tl.exp2(s - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)

        vb = v_ptr + tokens[:, None] * stride_v_token + vg
        v0 = tl.load(vb + (0 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v1 = tl.load(vb + (1 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v2 = tl.load(vb + (2 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        if PV_FP8:
            pb = (probabilities * P_SCALE).to(tl.float8e4nv)
        else:
            pb = probabilities.to(qr0.dtype)
            v0 = v0.to(qr0.dtype)
            v1 = v1.to(qr0.dtype)
            v2 = v2.to(qr0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(pb, v0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(pb, v1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(pb, v2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    # One reciprocal per head instead of a divide per output element, as in the hd192
    # prefill kernel. P_SCALE rides out here too, so neither it nor q_scale reaches the
    # output, and both -- like v_scale -- are constant per split and so commute with the
    # LSE-weighted merge.
    if PV_FP8:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum * P_SCALE), 0.0) * v_scale
    else:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum), 0.0) * v_scale
    output0 = acc0 * inv[:, None]
    output1 = acc1 * inv[:, None]
    output2 = acc2 * inv[:, None]
    oh = (group * heads_per_group + heads) * stride_out_head
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (0 * DK + dk)[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (1 * DK + dk)[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (2 * DK + dk)[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        # running_max/running_sum are base-2; the stored lse is natural log, as the
        # general kernel's is, so convert rather than changing the contract.
        tl.where(
            selected,
            (running_max + tl.log2(running_sum)) * 0.6931471805599453,
            NEGATIVE_INFINITY,
        ),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """One decode query against one KV group's selected regions, over one slot window.

    Structurally the same online softmax as prefill, but the metadata contract differs in
    the one way that matters: decode packs ``start_token | (phys_region << 32)`` in int64
    and does **not** carry ``valid_tokens``. The visible length is recomputed here as
    ``clamp(kv_seqlen - start_token, 0, region_size)``, which is why the logical start
    token has to be in the word at all -- the physical region alone cannot say how much of
    the last region exists yet.

    That also disposes of both padding sentinels without a special case. ``-1`` and
    ``0x7FFF_FFFF_FFFF_FFFF`` both leave ``0xFFFF_FFFF`` in the low word, so
    ``kv_seqlen - start_token`` is hugely negative and the slot clamps to zero tokens.

    ``v_scale``, and the ``k_scale`` the wrapper folded into ``softmax_scale``, dequantize
    an fp8 cache on the fp32 side. Both are per-split-safe: scaling every split's output
    by the same constant commutes with the LSE-weighted merge.
    """
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + request * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        # ``slots < window_end`` is not redundant: ``slots_per_split`` is a ceiling
        # division, so it is a multiple of BLOCK_R only by luck. The production
        # width 513 over 16 splits gives 33 slots per split against BLOCK_R=16, and
        # the last tile of every split would otherwise reach 15 slots into the next
        # split's window. Both splits then accumulate those regions and the merge
        # double-counts them -- the outputs stay finite and the split-local LSEs
        # stay plausible, so only a comparison against the unsplit reference sees it.
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        # ``physical`` is a sentinel-sized integer on dead slots, so it must be
        # neutralised before it becomes an address -- masking the load is not enough,
        # since the pointer arithmetic itself would overflow.
        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        keys = tl.load(
            k_ptr
            + tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        keys = keys.to(queries.dtype)
        values = values.to(queries.dtype)

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(
            token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY
        )

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = (
        tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0) * v_scale
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


@triton.jit
def _merge_split_states_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Combine per-split attention states into one, weighting by the partial lse.

    Each split produced an output already divided by its own softmax denominator, so the
    combination is a softmax over the partial log-sum-exps rather than a plain sum:
    ``out = sum_s softmax(lse)_s * out_s`` and ``lse = logsumexp_s(lse_s)``. A split that
    covered no live slot reports ``lse = -inf`` and drops out with weight zero; if *every*
    split does, the row selected nothing and must come back as a zero output with
    ``lse = -inf`` rather than ``0/0``.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, BLOCK_D)
    split_ok = splits < num_splits
    dim_ok = dims < head_dim

    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.exp(partial_lse - shift), 0.0)
    total = tl.sum(weights)

    states = tl.load(
        partial_out_ptr
        + splits[:, None] * stride_po_split
        + token * stride_po_token
        + head * stride_po_head
        + dims[None, :],
        mask=split_ok[:, None] & dim_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, None], axis=0) / tl.maximum(total, 1.0e-20)

    tl.store(
        out_ptr + token * stride_out_token + head * stride_out_head + dims,
        merged.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )
    tl.store(
        lse_ptr + token * stride_lse_token + head,
        tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
    )


def merge_split_states(
    partial_out: torch.Tensor, partial_lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[splits, tokens, heads, dim]`` partial states to ``[tokens, heads, dim]``.

    Semantics follow ``merge_variable_split_nat_lse_states_sm90_gqa`` -- "nat" being the
    natural-log lse the sparse kernels emit. The upstream kernel additionally supports a
    *variable* split plan, where the first ``n_split4`` requests are split four ways, the
    next ``n_split2`` twice, and the rest not at all, packed contiguously. That layout
    exists to fill the SMs at small batch and carries no semantics, so the release uses a
    uniform split and pays a little occupancy for a much simpler contract.
    """
    num_splits, tokens, num_heads, head_dim = partial_out.shape
    out = torch.empty(
        (tokens, num_heads, head_dim),
        device=partial_out.device,
        dtype=partial_out.dtype,
    )
    lse = torch.empty(
        (tokens, num_heads), device=partial_out.device, dtype=torch.float32
    )
    _merge_split_states_kernel[(tokens, num_heads)](
        partial_out,
        partial_lse,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        num_splits,
        head_dim=head_dim,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out, lse


def sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    num_splits: int = 1,
    block_regions: int = 16,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    q_scale: float = 1.0,
    qk_fp8: bool = False,
    pv_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention for one query token per request.

    Args:
        query: ``[num_reqs, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, pages flattened.
            May hold ``float8_e4m3``; see ``k_scale``/``v_scale``.
        packed_regions: ``[num_kv_groups * num_reqs, topk]`` **int64**, each slot
            ``start_token | (phys_region << 32)``. Note this is a different packing from
            prefill's int32 ``phys_region | (valid_tokens << 24)``: decode carries the
            logical start token and derives the visible length from ``kv_seqlens``.
        region_counts: ``[num_kv_groups * num_reqs]`` int32, slots to read per row.
        kv_seqlens: ``[num_reqs]`` int32, context length including the current token.
        num_splits: split the slot window this many ways and return partial states. With
            ``num_splits == 1`` the returned tensors are already the final result;
            otherwise pass them to :func:`merge_split_states`.
        block_regions: region-tile height for the general kernel. The ``head_dim=192``
            path ignores it and uses its own measured tile; see the launch there.
        k_scale, v_scale: per-tensor dequantization scales for an fp8 cache, as in
            :func:`sparse_attention_prefill`. Safe under splitting: a constant factor on
            every partial output survives the LSE-weighted merge unchanged.
        q_scale: prescale applied to the query before it is quantized to fp8, when
            ``qk_fp8`` is on. It only has to keep the query inside e4m3's range (2**-9
            subnormal to 448); Step-4 queries sit near 1 after the norm, so the default
            1.0 is right. Divided back out of ``softmax_scale``, so like
            ``k_scale``/``v_scale`` it never reaches the output and commutes with the
            split merge.
        qk_fp8: on the ``head_dim=192`` path with a ``float8_e4m3fn`` key cache, quantize
            the query to e4m3 and feed both operands of the QK dot in as fp8, which is the
            only way to reach the hardware fp8 MMA -- roughly 3x on the production decode
            shape. **Off by default, and deliberately not implied by an fp8 cache.** The
            fp8 cache alone is still a bf16-grade operator, because the kernel and the
            reference read the same fp8 bytes and the storage error cancels; quantizing
            the query is new error that no fp32 reference models, and it lands outside
            stage 6's bf16 envelope (see ``operator_parity/README.md``). Turning it on is
            an accuracy decision that needs an end-to-end score, not a kernel decision.
        pv_fp8: additionally quantize the probabilities and keep V in fp8 for the second
            dot. Requires ``qk_fp8``. Faster again (the fp8 path drops below even a bf16
            cache), and the accuracy cost of the whole fp8 arithmetic path is
            concentrated here -- roughly 5x what ``qk_fp8`` alone adds. Same caveat,
            more of it.

    Returns:
        ``num_splits == 1``: ``(out [num_reqs, num_heads, head_dim], lse [num_reqs, num_heads])``.
        Otherwise the same with a leading ``num_splits`` axis.

    As in prefill, the selected regions must be distinct. The upstream decode reference is
    stricter about this than the prefill one: it concatenates the gathered spans with no
    dedup at all, so a duplicated region would be counted twice in the softmax.
    """
    num_reqs, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * num_reqs:
        raise ValueError(
            f"expected {num_kv_groups * num_reqs} metadata rows, got {packed_regions.shape[0]}"
        )
    if packed_regions.dtype != torch.int64:
        raise ValueError(f"decode metadata must be int64, got {packed_regions.dtype}")
    if kv_seqlens.shape[0] != num_reqs:
        raise ValueError(f"expected {num_reqs} kv_seqlens, got {kv_seqlens.shape[0]}")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    scale = scale * k_scale
    # Every split walks the same number of slots so the window bound is a constexpr; the
    # tail split simply finds every slot masked out.
    slots_per_split = -(-topk // num_splits)

    out = torch.empty(
        (num_splits, num_reqs, num_heads, head_dim),
        device=query.device,
        dtype=query.dtype,
    )
    lse = torch.empty(
        (num_splits, num_reqs, num_heads), device=query.device, dtype=torch.float32
    )

    if head_dim == 192 and query.dtype in (torch.bfloat16, torch.float16):
        # Only OCP e4m3 goes into the dot as fp8. The AMD-native ``float8_e4m3fnuz`` is
        # rejected by the Triton backend here, and reinterpreting its bytes as e4m3fn
        # would be off by an exponent bias, so refuse rather than quietly produce a
        # plausible wrong answer.
        if qk_fp8 and key_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"qk_fp8 needs a float8_e4m3fn key cache, got {key_cache.dtype}"
            )
        if pv_fp8:
            if not qk_fp8:
                raise ValueError(
                    "pv_fp8 requires qk_fp8; alone it is slower and lossier"
                )
            if value_cache.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"pv_fp8 needs a float8_e4m3fn value cache, got {value_cache.dtype}"
                )
        # Launch config, measured on the three production decode shapes (b8/65K,
        # b4/32K, b1/32K) against the previous ``num_warps=4, BLOCK_R=16, stages=1``.
        # Two warps over a half-width region tile wins in every mode -- eight regions
        # is 64 tokens, still a full MMA tile at DK=64, and the narrower tile is what
        # lets two of them stage. Only the stage count splits by code path.
        hd192_block_regions = _fit_block_regions(
            min(8, triton.next_power_of_2(slots_per_split)),
            region_size=region_size,
            block_d=64,
            element_size=query.element_size(),
        )
        # The widen path -- fp8 cache read into a bf16 dot -- unpacks every loaded byte
        # in software, and that ALU work already covers the load latency that double
        # buffering exists to hide, so a second stage only costs registers there (1.41x
        # at one stage, 1.06x at two). Every other mode feeds the dot directly and
        # prefers two. A device too small to stage eight regions drops back to one.
        widening_fp8 = key_cache.dtype == torch.float8_e4m3fn and not qk_fp8
        hd192_stages = 1 if widening_fp8 or hd192_block_regions < 8 else 2
        _sparse_attn_decode_hd192_kernel[(num_reqs, num_kv_groups, num_splits)](
            query,
            key_cache,
            value_cache,
            packed_regions,
            region_counts,
            kv_seqlens,
            out,
            lse,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            packed_regions.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            # Base-2 scale: the kernel's online softmax runs on exp2, as hd192 prefill's
            # does, and converts back to a natural-log lse on the way out.
            (scale / q_scale if qk_fp8 else scale) * 1.4426950408889634,
            v_scale,
            q_scale,
            num_reqs,
            topk=topk,
            heads_per_group=heads_per_group,
            region_size=region_size,
            slots_per_split=slots_per_split,
            BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
            BLOCK_R=hd192_block_regions,
            DK=64,
            QK_FP8=qk_fp8,
            PV_FP8=pv_fp8,
            # e4m3 bottoms out at 2**-9 subnormal and post-max-subtraction
            # probabilities sit below that, so P is lifted before the cast. Measured
            # indifferent between 1 and 256 on the production shape -- the exponent
            # headroom above is what matters, and there is plenty.
            P_SCALE=256.0,
            num_warps=2,
            num_stages=hd192_stages,
        )
        if num_splits == 1:
            return out[0], lse[0]
        return out, lse

    block_d = triton.next_power_of_2(head_dim)
    _sparse_attn_decode_kernel[(num_reqs, num_kv_groups, num_splits)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        scale,
        v_scale,
        num_reqs,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_D=block_d,
        BLOCK_R=_fit_block_regions(
            min(block_regions, triton.next_power_of_2(slots_per_split)),
            region_size=region_size,
            block_d=block_d,
            # Compute dtype, not cache dtype -- see sparse_attention_prefill.
            element_size=query.element_size(),
        ),
    )
    if num_splits == 1:
        return out[0], lse[0]
    return out, lse


assert DECODE_PHYS_SHIFT.value == _MERGED_DECODE_PHYS_SHIFT.value == 32, (
    "merged DSA modules disagree on DECODE_PHYS_SHIFT"
)
del _MERGED_DECODE_PHYS_SHIFT


# ============================================================================
# merged source: inference/dsa_attention.py
# ============================================================================
"""DSA plumbing: the cache that persists a selection's inputs, and the scoring that uses it.

The kernels in this package are stateless -- each one takes the whole context as a tensor.
Generation is not: a decode step sees one token and has to score it against a context it no
longer holds. This module is the state that closes that gap, and it holds exactly three
things per DSA layer.

**The main K/V, in physical token order.** Written where the block table says, read by the
attention kernels. Nothing here is logical.

**One compressed summary per *completed* region, in logical order.** This is what the
indexer scores against, and "completed" is the load-bearing word. A region's summary is a
softmax-weighted mean over its tokens, so a summary computed from a partial region is a
different vector than the one computed from the full region -- storing the partial value
would silently poison every later selection. It is safe to only ever store completed
summaries because the selector's candidate range stops at ``q_position // region_size``:
every region it can choose ended before the query's own region began, so every candidate is
complete by construction. The query's own region is force-appended by the metadata builder
and never scored.

**The tokens of the region currently being filled.** At most ``region_size - 1`` of them.
When the region fills, its summary is computed with the same kernel the prefill path uses,
from the same inputs -- so the streaming result is not merely close to the batch result, it
is the same call. The alternative, carrying a running fp32 accumulator, is also exact but
has to reproduce the reduction order to stay that way.

Batching is by block table: request ``b``'s logical page ``p`` is physical page
``b * pages_per_request + p``, which makes the flat token space a reshaped
``[batch, max_tokens, ...]`` and the region space a reshaped ``[batch, max_regions, ...]``.
Scoring is looped over requests because each one scores against a different key set; the
attention itself is a single batched launch.
"""


import math
from dataclasses import dataclass


def build_rope_cache(
    *,
    rotary_span: int,
    theta: float,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for a NEOX-style partial rotation.

    Args:
        rotary_span: how many of the head's dimensions rotate. The returned tables are
            ``[max_position, rotary_span // 2]`` -- *half* the span, because each frequency
            drives the pair ``(d, d + rotary_span // 2)``. The kernels take that halved
            width as their ``rotary_dim``, which is easy to misread as the span itself.
        scaling: the ``rope_scaling`` block, or ``None``. Only ``llama3`` is implemented,
            because it is the only type step4's config asks for; an unrecognised type
            raises rather than silently falling back to unscaled, which would look like a
            long-context quality regression rather than a bug.
    """
    if rotary_span % 2:
        raise ValueError(f"rotary span must be even, got {rotary_span}")
    half = rotary_span // 2
    # Match vLLM's RotaryEmbedding cache construction exactly.  In particular this must
    # stay float32 throughout, even though computing the table in float64 looks more
    # accurate in isolation.  The deployed model rounds the float32 frequencies/trig
    # results to the requested cache dtype; using float64 changes BF16 table entries as
    # early as position 24 (and therefore changes Q/K before the first long-context
    # feature is involved).
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_span, 2, device=device, dtype=torch.float32)
            / rotary_span
        )
    )
    if scaling:
        inv_freq = _apply_llama3_scaling(inv_freq, scaling)

    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    assert freqs.shape[1] == half
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    rope_type = str(scaling.get("rope_type", scaling.get("type", ""))).lower()
    if rope_type != "llama3":
        raise NotImplementedError(f"unsupported rope_scaling type {rope_type!r}")

    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max = float(scaling["original_max_position_embeddings"])

    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths are stretched by the full factor, short ones are left alone, and the
    # band between is interpolated. Applying the stretch everywhere is the classic mistake:
    # it destroys the high-frequency positional detail the short-range heads rely on.
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    in_band = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(in_band, smoothed, scaled)


@dataclass(frozen=True)
class DSAGeometry:
    """The region/page arithmetic, in one place.

    ``regions_per_page`` is derived rather than configured: it is the ratio the metadata
    kernels use to split a logical region id into a page lookup and an offset, so a
    configured value that disagreed with ``page_size // region_size`` would address the
    wrong tokens.
    """

    proxy_dim: int
    topk: int
    region_size: int = REGION_BLOCK_SIZE
    page_size: int = 16

    @property
    def regions_per_page(self) -> int:
        if self.page_size % self.region_size:
            raise ValueError(
                f"page size {self.page_size} must be a multiple of region size {self.region_size}"
            )
        return self.page_size // self.region_size


class DSALayerCache:
    """Per-layer DSA state for one batch of in-flight sequences.

    Allocated once for a whole generation, so ``max_tokens`` is a capacity rather than a
    length; ``lengths`` tracks how much of it is live. Every tensor is addressed by the
    conventions the module docstring lists, and mixing them up is the failure this class
    exists to prevent. ``num_kv_groups`` is the *local* count: it is one for the production
    TP=8 layout, with the corresponding global KV head selected while pre-sharding.
    """

    def __init__(
        self,
        *,
        batch: int,
        max_tokens: int,
        num_kv_groups: int,
        head_dim: int,
        geometry: DSAGeometry,
        device: torch.device,
        dtype: torch.dtype,
        kv_dtype: torch.dtype | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        pages = -(-max_tokens // geometry.page_size)
        self.geometry = geometry
        self.batch = batch
        # Rounded up to a whole page so a request's token span never straddles the boundary
        # between two requests' page ranges.
        self.max_tokens = pages * geometry.page_size
        self.max_regions = self.max_tokens // geometry.region_size

        # ``dtype`` stays the compute dtype; ``kv_dtype`` is what the main K/V is stored
        # as, which may be float8_e4m3 to halve the cache. The scales are per-tensor and
        # default to 1.0 -- the sliding-attention path's fp8 kernels are built with
        # FP8_NO_SCALES and silently treat every descale as 1.0, so anything else here
        # would make the two attention families disagree on what the cache bytes mean.
        self.kv_dtype = dtype if kv_dtype is None else kv_dtype
        self.k_scale = float(k_scale)
        self.v_scale = float(v_scale)

        self.key = torch.zeros(
            (batch * self.max_tokens, num_kv_groups, head_dim),
            device=device,
            dtype=self.kv_dtype,
        )
        self.value = torch.zeros_like(self.key)
        self.summary = torch.zeros(
            (batch * self.max_regions, 1, geometry.proxy_dim),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.pending_key = torch.zeros(
            (batch, geometry.region_size, 1, geometry.proxy_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.pending_z = torch.zeros_like(self.pending_key)
        self.lengths = torch.zeros((batch,), device=device, dtype=torch.int32)

        # Request b owns pages [b * pages, (b + 1) * pages), so the mapping is an offset
        # identity. Materialising it as a real table rather than special-casing the
        # contiguous layout keeps the one code path the metadata gates actually cover.
        self.block_table = (
            torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
            + torch.arange(batch, device=device, dtype=torch.int32).view(-1, 1) * pages
        ).contiguous()

    def token_slice(self, request: int, start: int, count: int) -> slice:
        base = request * self.max_tokens + start
        return slice(base, base + count)

    def write_kv(
        self, request: int, start: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Store ``key``/``value`` for one request's tokens at logical position ``start``."""
        span = self.token_slice(request, start, key.shape[0])
        self.key[span] = self._store_dtype(key, self.k_scale)
        self.value[span] = self._store_dtype(value, self.v_scale)

    def _store_dtype(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Convert to the cache dtype, quantizing when that dtype is fp8.

        A plain ``.to(float8_e4m3fn)`` is not enough: e4m3 saturates at 448 and torch
        clamps silently, so every value above it would land on the same number. Dividing
        by the scale first is the division the kernels undo, and the explicit clamp keeps
        an out-of-range activation a bounded error rather than an inf.
        """
        if self.kv_dtype != torch.float8_e4m3fn:
            return tensor
        scaled = tensor.float() if scale == 1.0 else tensor.float() / scale
        return scaled.clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


class Step4SparseIndexer:
    """The indexer's projections and scoring, borrowing its parameters from the attention.

    Not an ``nn.Module``: the checkpoint stores the indexer weights directly under
    ``self_attn`` (``self_attn.sparse_indexer_q.weight``), so wrapping them in a submodule
    would rename every key and require a conversion step to load. The upstream
    implementation splits the same way and for the same reason.
    """

    def __init__(self, owner, *, geometry: DSAGeometry, num_kv_groups: int) -> None:
        self.owner = owner
        self.geometry = geometry
        # Provider groups and attention KV groups are the same partition.  Under TP they
        # are both local counts; for TP=8, one provider group (four indexer heads) drives
        # one local KV group (eight attention Q heads).
        self.num_kv_groups = num_kv_groups
        self.num_heads = int(owner.sparse_indexer_w.weight.shape[0])
        q_out = int(owner.sparse_indexer_q.weight.shape[0])
        if q_out != self.num_heads * geometry.proxy_dim:
            raise ValueError(
                "local sparse_indexer_q/w shapes disagree: "
                f"q rows={q_out}, w heads={self.num_heads}, proxy_dim={geometry.proxy_dim}. "
                "This commonly means stale TP shards from the old replicated-provider layout."
            )
        if self.num_heads % num_kv_groups:
            raise ValueError(
                f"{self.num_heads} indexer heads do not divide into {num_kv_groups} groups"
            )
        self.heads_per_group = self.num_heads // num_kv_groups
        # The indexer's groups *are* the attention's KV groups, so this ratio is fixed under
        # every tensor-parallel degree and the prescale below is a TP invariant.
        self.weight_prescale = float(self.heads_per_group) ** -0.5

    def project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and normalise the indexer's q/k/z, and scale its per-head weights.

        Returns:
            ``(index_q [tokens, groups, heads_per_group, proxy], index_k [tokens, 1, proxy],
            index_z [tokens, 1, proxy], weights [tokens, groups, heads_per_group] fp32)``.
        """

        owner = self.owner
        proxy_dim = self.geometry.proxy_dim
        tokens = hidden_states.shape[0]

        index_q = owner.sparse_indexer_q(hidden_states)
        index_k = owner.sparse_indexer_k(hidden_states)
        index_z = owner.sparse_indexer_z(hidden_states)

        index_q, index_k, index_z = indexer_norm_rope(
            index_q.contiguous(),
            index_k.contiguous(),
            index_z.contiguous(),
            # The factory checkpoint stores these norm parameters in FP32, while
            # vLLM constructs the runtime modules under the model's BF16 default
            # dtype and therefore rounds them during load.  Keep FP32 registered
            # parameters for a lossless checkpoint load, but reproduce that BF16
            # runtime boundary before the fused kernel consumes them.
            owner.sparse_indexer_q_norm.weight.to(dtype=index_q.dtype),
            owner.sparse_indexer_k_norm.weight.to(dtype=index_k.dtype),
            owner.sparse_indexer_k_norm.bias.to(dtype=index_k.dtype),
            owner.sparse_indexer_rope_cos,
            owner.sparse_indexer_rope_sin,
            positions,
            head_dim=proxy_dim,
            num_q_heads=self.num_heads,
            num_k_heads=1,
            rotary_dim=owner.sparse_indexer_rope_cos.shape[1],
            eps=owner.sparse_indexer_q_norm.variance_epsilon,
            q_norm_weight_bias=1.0,
        )

        # The checkpoint stores this tiny matrix in fp32, but the NV reference does not run
        # an fp32 GEMM with it. ``Step3p5SparseIndexerIndexTPLinear`` allocates the runtime
        # parameter with the attention projection's BF16 ``params_dtype`` and its loader
        # copies (therefore rounds) the fp32 checkpoint tensor into that parameter; only the
        # linear result is widened to fp32 before the prescale.  See
        # ``origin/dev/sparse_attn`` step3p5.py:887-892 and 1291-1336.  Keep our registered
        # parameter fp32 so the factory checkpoint loads without conversion, but reproduce
        # the deployed arithmetic by rounding W to the activation dtype before the GEMM.
        weights = torch.nn.functional.linear(
            hidden_states,
            owner.sparse_indexer_w.weight.to(dtype=hidden_states.dtype),
        ).float()
        weights = weights.view(tokens, self.num_kv_groups, self.heads_per_group)
        weights = weights * self.weight_prescale

        return (
            index_q.view(tokens, self.num_kv_groups, self.heads_per_group, proxy_dim),
            index_k.view(tokens, 1, proxy_dim),
            index_z.view(tokens, 1, proxy_dim),
            weights.contiguous(),
        )


def update_summaries_prefill(
    cache: DSALayerCache,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Compress a packed prefill into per-region summaries, buffering the ragged tail.

    Only regions that are *complete* get a summary. The tail tokens are held instead, and
    the region they belong to is summarised later, once it fills -- see the module
    docstring for why a partial summary must never be stored.
    """
    geometry = cache.geometry
    region_size = geometry.region_size
    starts, counts = csa_region_layout(
        seq_lens, regions_per_seq=cache.max_regions, region_size=region_size
    )
    # A count below ``region_size`` marks the ragged tail; zero makes the kernel skip it.
    counts = torch.where(counts == region_size, counts, torch.zeros_like(counts))

    _, summary_fp8 = csa_compress_regions(
        index_k, index_z, starts, counts, region_size=region_size
    )
    live = counts > 0
    cache.summary[live] = summary_fp8[live]

    token_starts = torch.cumsum(seq_lens.to(torch.int64), dim=0) - seq_lens.to(
        torch.int64
    )
    for request in range(cache.batch):
        length = int(seq_lens[request])
        tail = length % region_size
        cache.pending_key[request, :tail] = index_k[
            int(token_starts[request]) + length - tail : int(token_starts[request])
            + length
        ].to(cache.pending_key.dtype)
        cache.pending_z[request, :tail] = index_z[
            int(token_starts[request]) + length - tail : int(token_starts[request])
            + length
        ].to(cache.pending_z.dtype)


def update_summaries_decode(
    cache: DSALayerCache, index_k: torch.Tensor, index_z: torch.Tensor
) -> None:
    """Append one token per request and summarise any region that just filled.

    ``cache.lengths`` must already have been advanced past the new token, so the pending
    slot is derived from it rather than tracked separately -- one counter that can be wrong
    instead of two that can disagree.
    """
    geometry = cache.geometry
    region_size = geometry.region_size
    lengths = cache.lengths.tolist()

    for request in range(cache.batch):
        length = int(lengths[request])
        slot = (length - 1) % region_size
        cache.pending_key[request, slot] = index_k[request].to(cache.pending_key.dtype)
        cache.pending_z[request, slot] = index_z[request].to(cache.pending_z.dtype)
        if slot != region_size - 1:
            continue

        region = (length - 1) // region_size
        starts = torch.zeros((1,), device=index_k.device, dtype=torch.int32)
        counts = torch.full((1,), region_size, device=index_k.device, dtype=torch.int32)
        _, summary_fp8 = csa_compress_regions(
            cache.pending_key[request].contiguous(),
            cache.pending_z[request].contiguous(),
            starts,
            counts,
            region_size=region_size,
        )
        cache.summary[request * cache.max_regions + region] = summary_fp8[0]


def score_regions(
    indexer: Step4SparseIndexer,
    cache: DSALayerCache,
    index_q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    past_lens: torch.Tensor,
) -> torch.Tensor:
    """Indexer scores for every query against its own request's region summaries.

    Returns ``[groups * total_q, width]`` float32 in the group-major row order the selector
    requires, where ``width`` is the largest number of regions any row can see. Rows whose own
    history is shorter are left at zero past their bound; that is safe only because the
    selector's candidate range is bounded per row by its own history length and never reads
    further.

    ``width`` is computed rather than taken as ``cache.max_regions``, and the difference is not
    cosmetic. ``max_regions`` is the cache's *capacity*: at a 512k context it is 65536, so a
    4k-token prefill would allocate ``4 groups x 4096 queries x 65536`` fp32 -- 4.3 GB, for a
    tensor whose live corner is a thousandth of that. Sizing to the live width makes the
    allocation track the context rather than the reservation.

    The loop is over requests, not a batched GEMM, because each request scores against a
    different key set. At decode that is one tiny matmul per request; at prefill it is dwarfed
    by the attention it feeds.
    """
    groups = indexer.num_kv_groups
    total_q = int(index_q.shape[0])
    region_size = cache.geometry.region_size

    # A row can see every region completed before its query, so the widest row is the one with
    # the largest final position.
    last_positions = [
        int(past_lens[request]) + int(seq_lens[request]) - 1
        for request in range(cache.batch)
        if int(seq_lens[request]) > 0
    ]
    width = max((position // region_size for position in last_positions), default=0)
    logits = torch.zeros(
        (groups * total_q, max(width, 1)), device=index_q.device, dtype=torch.float32
    )

    offset = 0
    for request in range(cache.batch):
        length = int(seq_lens[request])
        if length == 0:
            continue
        regions = (int(past_lens[request]) + length - 1) // region_size
        if regions == 0:
            offset += length
            continue

        base = request * cache.max_regions
        # Scored straight into ``logits``: the destination row for group ``g`` and
        # local query ``q`` is ``g * total_q + offset + q``, which is exactly what
        # ``out_group_stride``/``out_row_base`` express. The summary pool is
        # e4m3-typed, so the key-side rounding would be the identity.
        indexer_logits(
            index_q[offset : offset + length].contiguous(),
            weights[offset : offset + length].contiguous(),
            cache.summary[base : base + regions].to(index_q.dtype),
            out=logits,
            out_row_base=offset,
            out_group_stride=total_q,
            quantize_keys=False,
        )
        offset += length

    return logits


def prefill_metadata(
    indexer: Step4SparseIndexer,
    cache: DSALayerCache,
    index_q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    past_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select, and pack prefill metadata for a packed batch."""
    groups = indexer.num_kv_groups
    device = index_q.device

    logits = score_regions(indexer, cache, index_q, weights, seq_lens, past_lens)
    positions, requests = _packed_positions(seq_lens, past_lens, device)
    return prefill_sparse_meta(
        logits,
        positions.repeat(groups).contiguous(),
        cache.block_table,
        requests.repeat(groups).contiguous(),
        topk=indexer.geometry.topk,
        region_size=indexer.geometry.region_size,
        regions_per_page=indexer.geometry.regions_per_page,
    )


def decode_metadata(
    indexer: Step4SparseIndexer,
    cache: DSALayerCache,
    index_q: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select, and pack decode metadata for one token per request."""
    ones = torch.ones((cache.batch,), device=index_q.device, dtype=torch.int32)
    logits = score_regions(
        indexer, cache, index_q, weights, ones, (cache.lengths - 1).clamp_min(0)
    )
    return decode_sparse_meta(
        logits,
        cache.lengths,
        cache.block_table,
        topk=indexer.geometry.topk,
        region_size=indexer.geometry.region_size,
        regions_per_page=indexer.geometry.regions_per_page,
    )


def _packed_positions(
    seq_lens: torch.Tensor, past_lens: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute position and owning request of every token in a packed batch."""
    positions = []
    requests = []
    for request in range(seq_lens.shape[0]):
        length = int(seq_lens[request])
        past = int(past_lens[request])
        positions.append(
            torch.arange(past, past + length, device=device, dtype=torch.int32)
        )
        requests.append(
            torch.full((length,), request, device=device, dtype=torch.int32)
        )
    return torch.cat(positions), torch.cat(requests)


# ============================================================================
# merged source: inference/fp8_gemm.py
# ============================================================================
"""Block-scaled fp8 GEMM for the MoE experts.

Adapted from ``deepseek-ai/DeepSeek-V4-Flash/inference/kernel.py`` (MIT,
Copyright (c) 2023 DeepSeek). Only the fp8 ``act_quant`` + ``fp8_gemm`` paths are
kept; the fp4, sinkhorn, and attention kernels in that file are intentionally
not imported -- this is the one piece the MoE experts need, and a smaller
surface is easier to audit.

Two facts about the DeepSeek scheme are easy to misimplement and invisible when
you do -- they are stated up front because neither is guessable from the
shapes alone:

* **Activations and weights use different blocking.** Activations are
  quantized dynamically at runtime in **1-D blocks of 128 along K** (the
  reduction dim), one fp32 scale per row per 128-K block. Weights are
  quantized offline in **2-D blocks of [128, 128] over (out, K)**, one fp32
  scale per block. The kernel below special-cases ``BLOCK_N == GROUP_N`` and
  ``BLOCK_K == GROUP_K`` so one tile sits inside exactly one weight block and
  one K-block; this is what lets a single weight scale (a scalar per K-iter)
  apply to the whole tile.

* **The stored weight scale uses the *inverse* convention.** The checkpoint
  name is ``weight_scale_inv`` because to dequantize you *multiply* by it,
  not divide: ``real_weight = w_fp8 * weight_scale_inv[block]``. The
  activation scale is the same kind of value (the divisor used during
  quantization); the names differ for historical reasons but both multiply
  back to the real value. The kernel folds the two scales into the fp32
  accumulator after the fp8 dot, never into the fp8 operands.

The dispatch between the fp8 path and the BF16 fallback (layers 88/89/90 ship
BF16 experts with no scale tensor) is by **presence of the scale tensor**,
never by layer index -- a layer can graduate off ``modules_to_not_convert``
without code changes here, and a grouped caller can pass ``None`` per-expert
for just the BF16 experts.
"""


import triton
import triton.language as tl

# fp8_e4m3fn has 4 exponent bits and 3 mantissa bits, with range [-448, 448].
# The activation quantizer maps each block's range into that envelope by
# dividing by ``max(amax, 1e-10) / 448``.  This is the Optimus/NV Step4
# ``per_token_group_quant_fp8`` default.  DeepSeek's standalone reference uses
# 1e-4, but that larger floor zeros low-amplitude blocks that the deployed
# Step4 path preserves.
FP8_E4M3_MAX: float = 448.0
FP8_AMAX_FLOOR: float = 1e-10

# K-block size shared by the activation quantizer and the weight scale grid.
# The kernel asserts BLOCK_K == GROUP_K so one K-iteration consumes exactly
# one act scale entry and one weight scale entry per row.
BLOCK_K: int = 128
# Weight N-block. The kernel asserts BLOCK_N == GROUP_N so one output tile
# sits inside exactly one weight N-block, keeping the weight scale a scalar
# per (tile, K-iter) instead of a vector.
BLOCK_N: int = 128


@triton.jit
def _act_quant_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Per-row 1-D block quantization to fp8_e4m3fn along K.

    One program handles a ``[BLOCK_M, BLOCK_K]`` tile -- exactly one quantization
    group wide. The amax is computed over the *live* elements of each row, so
    the kernel handles a final ragged K-block where the tile extends past K:
    the scale covers the live elements only, and the OOB quantized slots are
    not written (the GEMM's masked load will read them as zero, so the round
    trip stays numerically faithful).
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    tile_mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk,
        mask=tile_mask,
        other=0.0,
    ).to(tl.float32)

    # OOB elements are zero via ``other=0.0``, so they do not inflate the amax.
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-10)
    scale = amax * (1.0 / 448.0)
    inv_scale = 1.0 / scale

    # Clamp to [-448, 448] before the round-to-fp8: a value just above 448 would
    # otherwise wrap to NaN instead of saturating.
    q = tl.clamp(x * inv_scale[:, None], -448.0, 448.0)

    tl.store(
        y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yk,
        q.to(y_ptr.dtype.element_ty),
        mask=tile_mask,
    )
    tl.store(
        scale_ptr + rows * stride_sm + pid_k * stride_sk,
        scale,
        mask=row_mask,
    )


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization along the last dim of ``x``.

    Args:
        x: input tensor of shape ``[..., K]``. The last dim is the reduction
            dim K. Any leading dims are flattened; the result is reshaped back.
        block_size: K-block size, 128 by default. Must be a power of two so
            Triton accepts it as a constexpr bound on ``tl.arange``.

    Returns:
        ``(y, scale)`` where ``y`` has ``x``'s shape and dtype
        ``torch.float8_e4m3fn``, and ``scale`` has shape ``[..., ceil(K/block_size)]``
        and dtype ``torch.float32``. The recovery formula is
        ``real_x = y.float() * scale.unsqueeze(-1)``.

    Unlike the upstream DeepSeek kernel this does not assert ``K % block_size == 0``.
    A ragged last block is handled by computing the amax over the live elements
    only -- the MoE gate is exercised at K=4096 (a multiple of 128) in
    production, but the parity gate deliberately covers a ragged-K shape to
    keep the kernel honest.
    """
    if block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    leading = x.shape[:-1]
    K = x.shape[-1]
    x2 = x.contiguous()
    flat = x2.view(-1, K)
    M = flat.shape[0]
    y = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        M, (K + block_size - 1) // block_size, device=x.device, dtype=torch.float32
    )
    BLOCK_M = 16 if M < 32 else 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, block_size))
    _act_quant_kernel[grid](
        flat,
        y,
        scale,
        M,
        K,
        flat.stride(0),
        flat.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=block_size,
    )
    return y.view(*leading, K), scale.view(*leading, -1)


@triton.jit
def _fp8_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scale_ptr,
    W_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_wsn,
    stride_wsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
) -> None:
    """fp8 GEMM with per-128 block scaling on both operands.

    One program computes a ``[BLOCK_M, BLOCK_N]`` output tile. ``BLOCK_N`` must
    equal ``GROUP_N`` (128) so the tile sits inside exactly one weight N-block,
    and ``BLOCK_K`` must equal the activation block size (128) so one K-iter
    consumes exactly one act scale entry and one weight scale entry per row.

    Loads B transposed (as ``[BLOCK_K, BLOCK_N]``) so ``tl.dot`` sees ``A @ B``
    in the natural orientation. The strides are the *original* N/K strides of
    B; the transpose is by index, not by stride.

    The two scales are folded into the fp32 accumulator after the fp8 dot, not
    into the fp8 operands. Folding into the operands would re-quantize and
    lose the precision fp8 GEMM is supposed to keep; the post-dot multiply is
    the whole reason the accumulator is fp32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Weight scale N index: tile (pid_n) sits in weight N-block ``pid_n`` when
    # ``BLOCK_N == GROUP_N``. Tiles never span weight N-block boundaries.
    n_block = pid_n * BLOCK_N // GROUP_N

    k_iters = (K + BLOCK_K - 1) // BLOCK_K
    for k_idx in tl.range(0, k_iters):
        k_offset = k_idx * BLOCK_K
        k_mask = (k_offset + offs_k) < K

        a = tl.load(
            a_ptrs + k_offset * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs + k_offset * stride_bk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        # fp8 x fp8 -> fp32 on Hopper; this is the WGMMA path that makes the
        # scheme worth the precision cost. The accumulator is fp32 so the post-
        # dot scale multiply keeps full precision.
        partial = tl.dot(a, b, out_dtype=tl.float32)

        a_s = tl.load(
            A_scale_ptr + offs_m * stride_asm + k_idx * stride_ask,
            mask=m_mask,
            other=0.0,
        )
        # Scalar weight scale for this (n_block, k_idx) -- one fp32 per tile.
        w_s = tl.load(W_scale_ptr + n_block * stride_wsn + k_idx * stride_wsk)

        acc += partial * a_s[:, None] * w_s

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(C_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fp8_gemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 matmul ``y = x @ w.T``.

    Args:
        x_fp8: ``[M, K]`` ``torch.float8_e4m3fn`` activation, already quantized
            by :func:`act_quant`.
        x_scale: ``[M, ceil(K/128)]`` ``torch.float32`` activation scales.
        w_fp8: ``[N, K]`` ``torch.float8_e4m3fn`` weight (one expert's slice).
        w_scale_inv: ``[ceil(N/128), ceil(K/128)]`` ``torch.float32`` weight
            scales stored in the *inverse* convention
            (``real_weight = w_fp8 * w_scale_inv[block]``).

    Returns:
        ``y`` of shape ``[M, N]`` in ``torch.bfloat16``.
    """
    if x_fp8.dtype != torch.float8_e4m3fn or w_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("operands must be float8_e4m3fn")
    if x_scale.dtype != torch.float32 or w_scale_inv.dtype != torch.float32:
        raise TypeError("scales must be float32")
    if x_fp8.dim() != 2 or w_fp8.dim() != 2:
        raise ValueError("operands must be 2-D; group by expert at the call site")

    M, K = x_fp8.shape
    N, K_w = w_fp8.shape
    if K != K_w:
        raise ValueError(f"K mismatch: x has {K}, weight has {K_w}")
    k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    if x_scale.shape != (M, k_blocks):
        raise ValueError(f"x_scale shape {tuple(x_scale.shape)} != ({M}, {k_blocks})")
    if w_scale_inv.shape != ((N + BLOCK_N - 1) // BLOCK_N, k_blocks):
        raise ValueError(
            f"w_scale_inv shape {tuple(w_scale_inv.shape)} != "
            f"({(N + BLOCK_N - 1) // BLOCK_N}, {k_blocks})"
        )

    x_fp8 = x_fp8.contiguous()
    w_fp8 = w_fp8.contiguous()
    x_scale = x_scale.contiguous()
    w_scale_inv = w_scale_inv.contiguous()

    y = torch.empty((M, N), device=x_fp8.device, dtype=torch.bfloat16)
    BLOCK_M = 64 if M >= 64 else 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_gemm_kernel[grid](
        x_fp8,
        w_fp8,
        y,
        x_scale,
        w_scale_inv,
        M,
        N,
        K,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        y.stride(0),
        y.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale_inv.stride(0),
        w_scale_inv.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_N=BLOCK_N,
    )
    return y


def linear_fp8_or_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor | None,
) -> torch.Tensor:
    """``y = x @ weight.T`` with fp8 path when a scale is present, BF16 fallback.

    The dispatcher is the only place that decides fp8 vs BF16: layers 88/89/90
    ship BF16 experts with no scale, and the criterion is *presence of the
    scale tensor*, never the layer index. A layer can graduate off the
    not-convert list without code changes here.

    Args:
        x: ``[..., K]`` activation. Cast to BF16 internally on the fallback path
            (the fp8 path quantizes through BF16 anyway).
        weight: ``[N, K]`` weight; ``torch.float8_e4m3fn`` when
            ``weight_scale_inv`` is present, otherwise BF16.
        weight_scale_inv: ``[ceil(N/128), ceil(K/128)]`` fp32 inverse scales, or
            ``None`` to select the BF16 fallback path.

    Returns:
        ``[..., N]`` BF16 activation.
    """
    if weight_scale_inv is None:
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        x_bf = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        return torch.nn.functional.linear(x_bf, weight)

    leading = x.shape[:-1]
    K = x.shape[-1]
    x_bf = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    x_flat = x_bf.reshape(-1, K).contiguous()
    x_fp8, x_s = act_quant(x_flat)
    y = fp8_gemm(x_fp8, x_s, weight, weight_scale_inv)
    return y.view(*leading, -1)


# ============================================================================
# merged source: inference/moe_kernels.py
# ============================================================================
"""Small Triton primitives used by Step-4's MoE inference path.

The kernels in this module intentionally mirror the arithmetic of the deployed
vLLM/Optimus path rather than replacing it with mathematically equivalent
PyTorch reductions:

* SwiGLU loads the independent BF16 gate and up projections into FP32, applies
  the asymmetric clamps, multiplies in FP32, and rounds to BF16 only once.
* The top-k gather walks slots in logical order and maintains one FP32
  accumulator.  In particular, it is not a parallel reduction over the top-k
  dimension, whose reassociation could change the final BF16 value.

Importing this module does not initialize CUDA.  Triton is also an optional
import at module-import time so CPU-only tooling can inspect the modelling
code; attempting to execute either helper still requires CUDA and Triton.
"""


try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


_BLOCK_SIZE = 1024


if triton is not None:

    @triton.jit
    def _clamped_swiglu_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        num_elements,
        LIMIT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """FP32 SWIGLUstep over two independent, contiguous input tensors."""
        block = tl.program_id(0).to(tl.int64)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        gate_silu = gate * tl.sigmoid(gate)
        gate_clamped = tl.minimum(gate_silu, LIMIT)
        up_clamped = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        output = gate_clamped * up_clamped

        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _weighted_topk_gather_kernel(
        contributions_ptr,
        weights_ptr,
        output_ptr,
        hidden_size,
        TOP_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Sum one ``[top_k, hidden]`` slice in logical slot order."""
        token = tl.program_id(0).to(tl.int64)
        hidden_block = tl.program_id(1).to(tl.int64)
        hidden_offsets = hidden_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hidden_mask = hidden_offsets < hidden_size

        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        # TOP_K is constexpr, just as ``topk_num`` is in Optimus ep_gather.
        # Keeping the update in this loop produces the required slot-ordered
        # FP32 accumulation instead of a tree reduction over the K dimension.
        for slot in range(0, TOP_K):
            contribution_offset = (token * TOP_K + slot) * hidden_size + hidden_offsets
            contribution = tl.load(
                contributions_ptr + contribution_offset,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weights_ptr + token * TOP_K + slot).to(tl.float32)
            accumulator += contribution * weight

        output_offset = token * hidden_size + hidden_offsets
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )

else:  # Keep names available to importers and type/introspection tools.
    _clamped_swiglu_kernel = None
    _weighted_topk_gather_kernel = None


def _require_cuda_and_triton(tensors: tuple[torch.Tensor, ...]) -> None:
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Triton is required to execute Step-4 MoE kernels"
        ) from _TRITON_IMPORT_ERROR
    if any(tensor.device.type != "cuda" for tensor in tensors):
        devices = ", ".join(str(tensor.device) for tensor in tensors)
        raise RuntimeError(
            f"Step-4 MoE Triton kernels require CUDA tensors, got [{devices}]"
        )


def clamped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float = 7.0,
) -> torch.Tensor:
    """Apply Step-4's clamped SwiGLU to independent BF16 projections.

    Computes

    ``silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)``

    in FP32 and casts the product to the input dtype once.  ``gate`` and ``up``
    must have the same shape and be contiguous BF16 CUDA tensors.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate/up shape mismatch: {tuple(gate.shape)} != {tuple(up.shape)}"
        )
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise TypeError(
            f"gate and up must both be torch.bfloat16, got {gate.dtype} and {up.dtype}"
        )
    if gate.device != up.device:
        raise ValueError(f"gate/up device mismatch: {gate.device} != {up.device}")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must both be contiguous")

    try:
        limit_value = float(limit)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"limit must be a real scalar, got {limit!r}") from exc
    if not math.isfinite(limit_value) or limit_value < 0.0:
        raise ValueError(f"limit must be finite and non-negative, got {limit_value}")

    _require_cuda_and_triton((gate, up))
    output = torch.empty_like(gate)
    if gate.numel() == 0:
        return output

    assert triton is not None and _clamped_swiglu_kernel is not None
    grid = (triton.cdiv(gate.numel(), _BLOCK_SIZE),)
    _clamped_swiglu_kernel[grid](
        gate,
        up,
        output,
        gate.numel(),
        LIMIT=limit_value,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def weighted_topk_gather(
    contributions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weight and sum top-k contributions in deterministic slot order.

    Args:
        contributions: contiguous BF16 tensor of shape ``[T, K, H]``.
        weights: contiguous FP32 tensor of shape ``[T, K]``.

    Returns:
        Contiguous BF16 tensor of shape ``[T, H]``.  For every token and hidden
        element the kernel starts from FP32 zero and performs
        ``acc += contributions[:, slot].float() * weights[:, slot]`` for slots
        ``0, 1, ..., K - 1``, then casts the accumulator to BF16 once.
    """
    if contributions.ndim != 3:
        raise ValueError(
            f"contributions must have shape [T, K, H], got {tuple(contributions.shape)}"
        )
    if weights.ndim != 2:
        raise ValueError(f"weights must have shape [T, K], got {tuple(weights.shape)}")

    tokens, top_k, hidden_size = contributions.shape
    if weights.shape != (tokens, top_k):
        raise ValueError(f"weights shape {tuple(weights.shape)} != ({tokens}, {top_k})")
    if top_k <= 0:
        raise ValueError(f"top-k dimension must be positive, got {top_k}")
    if hidden_size <= 0:
        raise ValueError(f"hidden dimension must be positive, got {hidden_size}")
    if contributions.dtype != torch.bfloat16:
        raise TypeError(
            f"contributions must be torch.bfloat16, got {contributions.dtype}"
        )
    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be torch.float32, got {weights.dtype}")
    if contributions.device != weights.device:
        raise ValueError(
            "contributions/weights device mismatch: "
            f"{contributions.device} != {weights.device}"
        )
    if not contributions.is_contiguous() or not weights.is_contiguous():
        raise ValueError("contributions and weights must both be contiguous")

    _require_cuda_and_triton((contributions, weights))
    output = torch.empty(
        (tokens, hidden_size),
        dtype=torch.bfloat16,
        device=contributions.device,
    )
    if tokens == 0:
        return output

    assert triton is not None and _weighted_topk_gather_kernel is not None
    grid = (tokens, triton.cdiv(hidden_size, _BLOCK_SIZE))
    _weighted_topk_gather_kernel[grid](
        contributions,
        weights,
        output,
        hidden_size,
        TOP_K=top_k,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


# ============================================================================
# merged source: inference/qknorm_rope.py
# ============================================================================
"""Bitwise-oriented Triton Q/K RMSNorm + NeoX RoPE for Step-4.

The packed projection is laid out as ``[Q heads, K heads, V heads]``.  Q and K
are normalized head by head, rounded to BF16, reloaded as FP32, and only then
rotated.  V is copied without arithmetic.  Those details are observable: the
deployed CuTe kernel uses geometry-specific 16- or 32-lane RMS reductions and
has an activation-dtype materialization boundary between RMSNorm and RoPE.

This module intentionally does not depend on vLLM or CuTe.  Triton is optional
at import time so model/config tooling remains usable in CPU-only environments;
calling :func:`fused_qknorm_rope` still requires Triton and a CUDA device.
"""


import importlib

try:
    import triton
    import triton.language as tl

    # Triton moved libdevice from ``language`` to ``language.extra`` in 3.x.
    try:
        import triton.language.extra.libdevice as _libdevice
    except ImportError:  # pragma: no cover - exercised by the Triton 2.x runtime.
        _libdevice = tl.libdevice
except ImportError:  # pragma: no cover - covered by the CPU import contract.
    triton = None
    tl = None
    _libdevice = None


# CUDA exposes the explicitly rounded scalar intrinsics (``add_rn``, ``mul_rn``,
# ...); the HIP/DCU backend exposes no ``_rn``-suffixed symbol at all.  On HIP
# the plain fp32 operators are already round-to-nearest-even and bit-identical
# to the CUDA ``*_rn`` results, so the intrinsics' only remaining job is to stop
# the compiler contracting a separate multiply and add into one FMA.  That is
# instead handled at launch time by ``_NO_FP_FUSION`` below, which is why these
# fallbacks can be plain operators without loosening the numerical contract.
#
# ``fma`` stays an intrinsic: it must remain fused even with fusion disabled.
# Both backends provide it under that name.
#
# ``triton.language.extra.libdevice`` cannot be probed with ``hasattr``: it is a
# dispatcher that forwards every name to ``extra.<backend>.libdevice`` only at
# compile time, so it answers True for symbols the active backend lacks and the
# missing attribute surfaces as a CompilationError instead.  Resolve the backend
# module directly.  Anything unexpected (no driver, no device, an unknown
# backend) falls through to the plain-operator path, which is numerically
# correct on every backend because fusion is disabled alongside it.
def _detect_rounded_intrinsics() -> bool:
    if _libdevice is None:
        return False
    try:
        from triton.runtime.driver import driver

        backend = driver.active.get_current_target().backend
        module = importlib.import_module(f"triton.language.extra.{backend}.libdevice")
    except Exception:  # pragma: no cover - no driver or no device available.
        return False
    return all(
        hasattr(module, name)
        for name in ("add_rn", "sub_rn", "mul_rn", "div_rn", "fma_rn")
    )


_HAS_ROUNDED_INTRINSICS = _detect_rounded_intrinsics()

if triton is not None and not _HAS_ROUNDED_INTRINSICS:

    @triton.jit
    def _add_rn(a, b):
        return a + b

    @triton.jit
    def _sub_rn(a, b):
        return a - b

    @triton.jit
    def _mul_rn(a, b):
        return a * b

    @triton.jit
    def _div_rn(a, b):
        return a / b

    @triton.jit
    def _fma_rn(a, b, c):
        return _libdevice.fma(a, b, c)

elif triton is not None:

    @triton.jit
    def _add_rn(a, b):
        return _libdevice.add_rn(a, b)

    @triton.jit
    def _sub_rn(a, b):
        return _libdevice.sub_rn(a, b)

    @triton.jit
    def _mul_rn(a, b):
        return _libdevice.mul_rn(a, b)

    @triton.jit
    def _div_rn(a, b):
        return _libdevice.div_rn(a, b)

    @triton.jit
    def _fma_rn(a, b, c):
        return _libdevice.fma_rn(a, b, c)


# Passing ``enable_fp_fusion=False`` keeps ``a * b`` and ``+ c`` as two
# separately rounded operations, matching CUDA's ``mul_rn``/``add_rn`` pair,
# while explicit ``fma`` calls stay fused.  The CUDA path already gets this
# from the intrinsics themselves, so only ask for it when falling back.
_NO_FP_FUSION = {} if _HAS_ROUNDED_INTRINSICS else {"enable_fp_fusion": False}


HEAD_DIM = 192
_SUPPORTED_ROTARY_PAIRS = frozenset((32, 96))


if triton is not None:

    @triton.jit
    def _lane_square_sum_192(
        qkv_ptr,
        head_base,
        LANE: tl.constexpr,
    ):
        """CuTe's 32-pair layout: three four-value vectors for one of 16 lanes."""
        acc = 0.0
        # The order is part of the numerical contract:
        #   block 0 dims [0, 64), block 1 [64, 128), block 2 [128, 192),
        # with four adjacent values owned by each logical lane in every block.
        for block in range(3):
            for element in range(4):
                dim = block * 64 + LANE * 4 + element
                value = tl.load(qkv_ptr + head_base + dim).to(tl.float32)
                acc = _fma_rn(value, value, acc)
        return acc

    @triton.jit
    def _wide_lane_square_sum_192(
        qkv_ptr,
        head_base,
        LANE: tl.constexpr,
    ):
        """Installed vLLM CuTe layout for the 96-pair Step-4 path.

        Covering 96 RoPE pairs makes that kernel widen from 16 to 32 logical
        lanes.  Its 128-bit copy assigns eight adjacent values to each lane
        and pads the 192-value row to 256 values; lanes 24..31 therefore
        contribute eight explicit zeros apiece.
        """
        acc = 0.0
        for element in range(8):
            dim = LANE * 8 + element
            value = tl.load(
                qkv_ptr + head_base + dim,
                mask=dim < 192,
                other=0.0,
            ).to(tl.float32)
            acc = _fma_rn(value, value, acc)
        return acc

    @triton.jit
    def _qknorm_rope_192_kernel(
        qkv_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        stride_qkv_token,
        stride_q_out_token,
        stride_k_out_token,
        stride_cos_token,
        stride_sin_token,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        ROTARY_PAIRS: tl.constexpr,
        EPS: tl.constexpr,
        NORM_WEIGHT_BIAS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """One program per token and Q/K head.

        The physical Triton layout is deliberately not used to define the
        reduction.  Explicit logical lane sums followed by the exact ascending
        XOR tree make the result independent of Triton's evolving warp/layout
        lowering.  Step-4's 32-pair path uses 16 lanes; the 96-pair path uses
        32 so all rotary pairs are covered.
        """
        token = tl.program_id(0)
        packed_head = tl.program_id(1)
        is_q = packed_head < NUM_Q_HEADS
        head_base = token * stride_qkv_token + packed_head * 192

        if ROTARY_PAIRS == 96:
            # The production vLLM CuTe kernel widens to 32 lanes so four
            # RoPE values per lane cover all 96 pairs.  Its norm copy layout
            # changes at the same time to 8 contiguous values/lane, padded to
            # 256.  Preserve both the local FMA chains and XOR(1,2,4,8,16).
            p0 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=0)
            p1 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=1)
            p2 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=2)
            p3 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=3)
            p4 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=4)
            p5 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=5)
            p6 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=6)
            p7 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=7)
            p8 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=8)
            p9 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=9)
            p10 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=10)
            p11 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=11)
            p12 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=12)
            p13 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=13)
            p14 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=14)
            p15 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=15)
            p16 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=16)
            p17 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=17)
            p18 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=18)
            p19 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=19)
            p20 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=20)
            p21 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=21)
            p22 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=22)
            p23 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=23)
            p24 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=24)
            p25 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=25)
            p26 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=26)
            p27 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=27)
            p28 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=28)
            p29 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=29)
            p30 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=30)
            p31 = _wide_lane_square_sum_192(qkv_ptr, head_base, LANE=31)

            s01 = _add_rn(p0, p1)
            s23 = _add_rn(p2, p3)
            s45 = _add_rn(p4, p5)
            s67 = _add_rn(p6, p7)
            s89 = _add_rn(p8, p9)
            s1011 = _add_rn(p10, p11)
            s1213 = _add_rn(p12, p13)
            s1415 = _add_rn(p14, p15)
            s1617 = _add_rn(p16, p17)
            s1819 = _add_rn(p18, p19)
            s2021 = _add_rn(p20, p21)
            s2223 = _add_rn(p22, p23)
            s2425 = _add_rn(p24, p25)
            s2627 = _add_rn(p26, p27)
            s2829 = _add_rn(p28, p29)
            s3031 = _add_rn(p30, p31)
            s03 = _add_rn(s01, s23)
            s47 = _add_rn(s45, s67)
            s811 = _add_rn(s89, s1011)
            s1215 = _add_rn(s1213, s1415)
            s1619 = _add_rn(s1617, s1819)
            s2023 = _add_rn(s2021, s2223)
            s2427 = _add_rn(s2425, s2627)
            s2831 = _add_rn(s2829, s3031)
            s07 = _add_rn(s03, s47)
            s815 = _add_rn(s811, s1215)
            s1623 = _add_rn(s1619, s2023)
            s2431 = _add_rn(s2427, s2831)
            s015 = _add_rn(s07, s815)
            s1631 = _add_rn(s1623, s2431)
            sum_squares = _add_rn(s015, s1631)
        else:
            # 32-pair full-attention layers use 16 lanes.  Each pN is the
            # sequential 12-value fma.rn chain of logical lane N.
            p0 = _lane_square_sum_192(qkv_ptr, head_base, LANE=0)
            p1 = _lane_square_sum_192(qkv_ptr, head_base, LANE=1)
            p2 = _lane_square_sum_192(qkv_ptr, head_base, LANE=2)
            p3 = _lane_square_sum_192(qkv_ptr, head_base, LANE=3)
            p4 = _lane_square_sum_192(qkv_ptr, head_base, LANE=4)
            p5 = _lane_square_sum_192(qkv_ptr, head_base, LANE=5)
            p6 = _lane_square_sum_192(qkv_ptr, head_base, LANE=6)
            p7 = _lane_square_sum_192(qkv_ptr, head_base, LANE=7)
            p8 = _lane_square_sum_192(qkv_ptr, head_base, LANE=8)
            p9 = _lane_square_sum_192(qkv_ptr, head_base, LANE=9)
            p10 = _lane_square_sum_192(qkv_ptr, head_base, LANE=10)
            p11 = _lane_square_sum_192(qkv_ptr, head_base, LANE=11)
            p12 = _lane_square_sum_192(qkv_ptr, head_base, LANE=12)
            p13 = _lane_square_sum_192(qkv_ptr, head_base, LANE=13)
            p14 = _lane_square_sum_192(qkv_ptr, head_base, LANE=14)
            p15 = _lane_square_sum_192(qkv_ptr, head_base, LANE=15)

            # Lane 0 after ascending shfl_xor offsets 1, 2, 4, 8.
            s01 = _add_rn(p0, p1)
            s23 = _add_rn(p2, p3)
            s45 = _add_rn(p4, p5)
            s67 = _add_rn(p6, p7)
            s89 = _add_rn(p8, p9)
            s1011 = _add_rn(p10, p11)
            s1213 = _add_rn(p12, p13)
            s1415 = _add_rn(p14, p15)
            s03 = _add_rn(s01, s23)
            s47 = _add_rn(s45, s67)
            s811 = _add_rn(s89, s1011)
            s1215 = _add_rn(s1213, s1415)
            s07 = _add_rn(s03, s47)
            s815 = _add_rn(s811, s1215)
            sum_squares = _add_rn(s07, s815)

        mean_square = _div_rn(sum_squares, 192.0)
        inverse_rms = _libdevice.rsqrt(_add_rn(mean_square, EPS))

        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < 192
        values = tl.load(
            qkv_ptr + head_base + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        q_weight = tl.load(
            q_weight_ptr + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        k_weight = tl.load(
            k_weight_ptr + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.where(is_q, q_weight, k_weight)
        weight = _add_rn(weight, NORM_WEIGHT_BIAS)
        x_hat = _mul_rn(values, inverse_rms)
        normalized = _mul_rn(x_hat, weight).to(tl.bfloat16)

        q_base = token * stride_q_out_token + packed_head * 192
        k_base = token * stride_k_out_token + (packed_head - NUM_Q_HEADS) * 192
        tl.store(
            q_out_ptr + q_base + dims,
            normalized,
            mask=dim_mask & is_q,
        )
        tl.store(
            k_out_ptr + k_base + dims,
            normalized,
            mask=dim_mask & ~is_q,
        )

        # This global-memory round trip is the Triton equivalent of CuTe's
        # activation-typed shared-memory boundary.  A barrier prevents the
        # reload below from observing the pre-store contents.
        tl.debug_barrier()

        pairs = tl.arange(0, BLOCK_R)
        pair_mask = pairs < ROTARY_PAIRS
        dim0 = pairs
        dim1 = pairs + ROTARY_PAIRS
        q_value0 = tl.load(
            q_out_ptr + q_base + dim0,
            mask=pair_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        q_value1 = tl.load(
            q_out_ptr + q_base + dim1,
            mask=pair_mask & is_q,
            other=0.0,
        ).to(tl.float32)
        k_value0 = tl.load(
            k_out_ptr + k_base + dim0,
            mask=pair_mask & ~is_q,
            other=0.0,
        ).to(tl.float32)
        k_value1 = tl.load(
            k_out_ptr + k_base + dim1,
            mask=pair_mask & ~is_q,
            other=0.0,
        ).to(tl.float32)
        value0 = tl.where(is_q, q_value0, k_value0)
        value1 = tl.where(is_q, q_value1, k_value1)

        position = tl.load(positions_ptr + token)
        cos_value = tl.load(
            cos_ptr + position * stride_cos_token + pairs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        sin_value = tl.load(
            sin_ptr + position * stride_sin_token + pairs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        # CuTe PTX contract:
        #   rot0 = mul(value0, cos) - mul(value1, sin)
        #   rot1 = fma(value0, sin, mul(value1, cos))
        rotated0 = _sub_rn(
            _mul_rn(value0, cos_value),
            _mul_rn(value1, sin_value),
        ).to(tl.bfloat16)
        rotated1 = _fma_rn(
            value0,
            sin_value,
            _mul_rn(value1, cos_value),
        ).to(tl.bfloat16)

        tl.store(
            q_out_ptr + q_base + dim0,
            rotated0,
            mask=pair_mask & is_q,
        )
        tl.store(
            q_out_ptr + q_base + dim1,
            rotated1,
            mask=pair_mask & is_q,
        )
        tl.store(
            k_out_ptr + k_base + dim0,
            rotated0,
            mask=pair_mask & ~is_q,
        )
        tl.store(
            k_out_ptr + k_base + dim1,
            rotated1,
            mask=pair_mask & ~is_q,
        )

    @triton.jit
    def _copy_v_192_kernel(
        qkv_ptr,
        v_out_ptr,
        stride_qkv_token,
        stride_v_out_token,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)
        mask = dims < 192
        input_base = (
            token * stride_qkv_token + (NUM_Q_HEADS + NUM_KV_HEADS + head) * 192
        )
        output_base = token * stride_v_out_token + head * 192
        # BF16 load/store only: V must retain its original bits.
        value = tl.load(qkv_ptr + input_base + dims, mask=mask)
        tl.store(v_out_ptr + output_base + dims, value, mask=mask)


def _validate_qknorm_rope_inputs(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float,
    norm_weight_bias: float,
) -> None:
    tensors = {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos": cos,
        "sin": sin,
        "positions": positions,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if rotary_pairs not in _SUPPORTED_ROTARY_PAIRS:
        raise ValueError(
            "rotary_pairs must be 32 (full-attention layers) or 96 "
            f"(sliding-attention layers), got {rotary_pairs}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if not math.isfinite(norm_weight_bias):
        raise ValueError(f"norm_weight_bias must be finite, got {norm_weight_bias}")

    packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.ndim != 2 or tuple(qkv.shape) != (qkv.shape[0], packed_width):
        raise ValueError(
            f"qkv must have shape [tokens, {packed_width}], got {tuple(qkv.shape)}"
        )
    if qkv.stride(1) != 1:
        raise ValueError("qkv must be contiguous in its last dimension")
    if qkv.dtype != torch.bfloat16:
        raise TypeError(f"qkv must have dtype torch.bfloat16, got {qkv.dtype}")

    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.ndim != 1 or weight.numel() != head_dim:
            raise ValueError(
                f"{name} must have shape [{head_dim}], got {tuple(weight.shape)}"
            )
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {weight.dtype}")

    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(
            f"cos and sin must be 2D, got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must have equal shapes, got {tuple(cos.shape)} "
            f"and {tuple(sin.shape)}"
        )
    if cos.shape[1] < rotary_pairs:
        raise ValueError(
            f"cos/sin width must be at least {rotary_pairs}, got {cos.shape[1]}"
        )
    if cos.stride(1) != 1 or sin.stride(1) != 1:
        raise ValueError("cos and sin must be contiguous in their last dimension")
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise TypeError(
            "cos and sin must have dtype torch.bfloat16, got "
            f"{cos.dtype} and {sin.dtype}"
        )

    if positions.ndim != 1 or positions.numel() != qkv.shape[0]:
        raise ValueError(
            "positions must have one entry per qkv token, got "
            f"shape {tuple(positions.shape)} for {qkv.shape[0]} tokens"
        )
    if not positions.is_contiguous():
        raise ValueError("positions must be contiguous")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "positions must have dtype torch.int32 or torch.int64, got "
            f"{positions.dtype}"
        )

    # Device checks intentionally come last.  Shape and dtype are properties of
    # the public contract too, and validating them first lets CPU-only tooling
    # diagnose malformed model inputs without needing a CUDA allocation.
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("all inputs must be on the same CUDA device")


def fused_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize and rotate packed TP-local Q/K heads, and copy V exactly.

    Args:
        qkv: BF16 ``[tokens, (num_q_heads + 2 * num_kv_heads) * 192]``.
            A wider physical row stride is allowed, but the last dimension must
            be contiguous.
        q_weight, k_weight: FP32 zero-centered RMSNorm weights of shape
            ``[192]``.  The effective scale is ``weight + norm_weight_bias``;
            Step-4 uses the default bias of ``1.0``.
        cos, sin: BF16 NeoX tables ``[max_position, >= rotary_pairs]``.
        positions: contiguous int32 or int64 position for every token.
        rotary_pairs: ``96`` on sliding layers and ``32`` on full-attention
            layers.  The leading ``2 * rotary_pairs`` dimensions are rotated.

    Returns:
        Flat contiguous BF16 tensors ``q``, ``k``, ``v`` with shapes
        ``[tokens, num_q_heads * 192]`` and
        ``[tokens, num_kv_heads * 192]`` (for both K and V).

    The older native BITWISE reference hard-codes four RoPE pairs per one of
    16 lanes, which only covers 64 pairs at D=192.  That is valid for its
    tested 48-pair geometry but incomplete for Step-4's 96-pair sliding
    layers.  The deployed CuTe/vLLM path widens that case to 32 lanes and a
    padded 256-value normalization tile.  This implementation reproduces that
    reduction and rotates all 96 pairs; 32-pair full-attention layers retain
    the 16-lane blocked layout.
    """
    _validate_qknorm_rope_inputs(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=eps,
        norm_weight_bias=norm_weight_bias,
    )
    if triton is None:
        raise RuntimeError("Triton is required to run fused_qknorm_rope")

    tokens = qkv.shape[0]
    q_out = torch.empty(
        (tokens, num_q_heads * HEAD_DIM),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    k_out = torch.empty(
        (tokens, num_kv_heads * HEAD_DIM),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    v_out = torch.empty_like(k_out)
    if tokens == 0:
        return q_out, k_out, v_out

    _qknorm_rope_192_kernel[(tokens, num_q_heads + num_kv_heads)](
        qkv,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos.stride(0),
        sin.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        ROTARY_PAIRS=rotary_pairs,
        EPS=float(eps),
        NORM_WEIGHT_BIAS=float(norm_weight_bias),
        BLOCK_D=256,
        BLOCK_R=128,
        num_warps=4,
        **_NO_FP_FUSION,
    )
    _copy_v_192_kernel[(tokens, num_kv_heads)](
        qkv,
        v_out,
        qkv.stride(0),
        v_out.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_D=256,
        num_warps=4,
    )
    return q_out, k_out, v_out


__all__ = [
    "REGION_BLOCK_SIZE",
    "PROXY_DIM",
    "shared_table_stride",
    "indexer_norm_rope",
    "csa_region_layout",
    "csa_compress_regions",
    "round_activations_e4m3",
    "indexer_logits",
    "region_topk_ids",
    "region_topk_pack",
    "prefill_sparse_meta",
    "decode_sparse_meta",
    "decode_rows_meta",
    "sparse_attention_prefill",
    "merge_split_states",
    "sparse_attention_decode",
    "build_rope_cache",
    "DSAGeometry",
    "DSALayerCache",
    "Step4SparseIndexer",
    "update_summaries_prefill",
    "update_summaries_decode",
    "score_regions",
    "prefill_metadata",
    "decode_metadata",
    "act_quant",
    "fp8_gemm",
    "linear_fp8_or_bf16",
    "clamped_swiglu",
    "weighted_topk_gather",
    "HEAD_DIM",
    "fused_qknorm_rope",
]
