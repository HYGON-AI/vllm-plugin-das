# SPDX-License-Identifier: MIT
"""Step5's regular channel-FP8 launch with signed-32-bit-safe A addressing.

Adapted from boltops.fused_moe.triton.moe_op (MIT). Reuse its compiled kernel;
only the large-activation launch lives here. The installed vendor module is
never changed. EP64's sorted A arena can exceed 2**31 one-byte elements before
it reaches the vendor dispatcher's unsigned-byte threshold of 2**32 - 1.
"""
import triton


def needs_wide_a(a):
    return a.numel() >= (1 << 31) or a.numel() * a.element_size() >= (1 << 32) - 1


def invoke_wide_fp8(a, b, c, a_scale, b_scale, weights, ids, sorted_ids,
                    sorted_weights, expert_ids, nvalid, top_k, compute_type,
                    config, *, c_sorted=False, a_sorted=False,
                    bottom_a_use_mls_load=False, ck_sorting=False, ck_topk=8,
                    scale_bias_with_routed_weight=False):
    """Regular W8A8/per-channel/no-bias path used by Step5's native EP backend."""
    from boltops.fused_moe.triton import moe_op

    cfg = dict(config)
    if cfg.pop("N_TILES_PER_CTA", 1) != 1 or cfg.get("SPLIT_K", 1) > 1:
        raise ValueError("Wide Step5 FP8 dispatch requires the regular single-N-tile kernel")
    cfg.pop("SPLIT_K", None)
    cfg.setdefault("USE_MLS_LOAD", False)
    if not moe_op.capMLS:
        cfg["USE_MLS_LOAD"] = False
    if bottom_a_use_mls_load:
        a_sorted = True
        bottom_a_use_mls_load = bool(moe_op.capMLS)
    bk = cfg.pop("BLOCK_SIZE_K")
    combine = cfg.pop("COMBINE_SCALE_LOAD", None)
    if combine and b.size(2) % (2 * bk):
        raise ValueError("COMBINE_SCALE_LOAD requires K divisible by twice BLOCK_SIZE_K")
    rows = sorted_ids.size(0)
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_SIZE_M'])
                         * triton.cdiv(b.size(1), meta['BLOCK_SIZE_N']),)
    byte_limit = (1 << 32) - 1
    moe_op.fused_moe_kernel[grid](
        a, b, c, weights, nvalid, expert_ids, sorted_ids, sorted_weights,
        a_scale, b_scale, b,
        b.size(1), b.size(2), rows, ids.numel(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
        c.stride(-2), c.stride(-1),
        a_scale.stride(0) if a_scale.ndim == 2 else 0,
        a_scale.stride(1) if a_scale.ndim == 2 else 0,
        b_scale.stride(0) if b_scale.ndim >= 2 else 0,
        b_scale.stride(2) if b_scale.ndim == 3 else 0,
        b_scale.stride(1) if b_scale.ndim >= 2 else 0,
        b.stride(0), b.stride(1), a.size(0), 0, 0,
        MUL_ROUTED_WEIGHT=False, USE_ADDR_OFFSET_INT64_A=True,
        USE_ADDR_OFFSET_INT64_B=b.numel() * b.element_size() >= byte_limit,
        USE_ADDR_OFFSET_INT64_C=c.numel() * c.element_size() >= byte_limit,
        top_k=top_k, compute_type=compute_type, use_fp8_w8a8=True,
        use_int8_w8a8=False, use_int8_w8a16=False, per_channel_quant=True,
        c_sorted=c_sorted, a_sorted=a_sorted,
        bottom_a_use_mls_load=bottom_a_use_mls_load,
        ck_sorting=ck_sorting, ck_topk=ck_topk, naive_block_assignment=False,
        NUM_XCDS=1, SCALE_BIAS_WITH_ROUTED_WEIGHT=scale_bias_with_routed_weight,
        ADD_BIAS=False, BLOCK_SIZE_K=bk, COMBINE_SCALE_LOAD=combine,
        N_TILES_PER_CTA=1, **cfg,
    )
