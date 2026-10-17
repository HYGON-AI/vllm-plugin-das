# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Rotate only the GPT-J suffix of a consumed BF16 attention output."""
import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_rope_suffix_inplace_kernel(
    # count scales with the token count: keep it runtime so each new batch
    # size does not trigger a ~200 ms JIT compile on the critical path.
    output, positions, cache, count, heads: tl.constexpr,
    token_stride: tl.constexpr, head_stride: tl.constexpr,
    cache_stride: tl.constexpr, nope: tl.constexpr, half: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair < count
    k = pair % half
    h = (pair // half) % heads
    t = pair // (half * heads)
    base = t * token_stride + h * head_stride + nope + 2 * k
    a = tl.load(output + base, mask=valid, other=0).to(tl.float32)
    b = tl.load(output + base + 1, mask=valid, other=0).to(tl.float32)
    pos = tl.load(positions + t, mask=valid, other=0).to(tl.int64)
    cos = tl.load(cache + pos * cache_stride + k, mask=valid, other=0)
    sin = tl.load(cache + pos * cache_stride + half + k, mask=valid, other=0)
    tl.store(output + base, (a * cos + b * sin).to(tl.bfloat16), mask=valid)
    tl.store(output + base + 1, (b * cos - a * sin).to(tl.bfloat16), mask=valid)


def inverse_rope_suffix_inplace(o, positions, cos_sin_cache, rope_head_dim):
    """Mutate the suffix only; caller must own and consume this output."""
    count = o.shape[0] * o.shape[1] * (rope_head_dim // 2)
    if count:
        _inverse_rope_suffix_inplace_kernel[(triton.cdiv(count, 256),)](
            o, positions, cos_sin_cache, count, o.shape[1],
            o.stride(0), o.stride(1), cos_sin_cache.stride(0),
            o.shape[-1] - rope_head_dim, rope_head_dim // 2,
            BLOCK=256, num_warps=4,
        )
    return o
