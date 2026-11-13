# SPDX-License-Identifier: Apache-2.0
"""Fused split for Qwen3.5 / Qwen3-Next GatedDeltaNet rearrange_mixed_qkv.

Replaces the three-way `torch.split(mixed_qkv, [K, K, V], dim=-1)` followed by
three `.contiguous()` copies with a single fused triton kernel that
scatter-writes each row of `mixed_qkv` to three separately-allocated
contiguous output buffers.

Original code (qwen3_next.py:656, inherited by qwen3_5):

    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = rearrange(q, "l (h d) -> 1 l h d", d=head_k_dim)   # view
    k = rearrange(k, "l (h d) -> 1 l h d", d=head_k_dim)   # view
    v = rearrange(v, "l (h d) -> 1 l h d", d=head_v_dim)   # view
    return q.contiguous(), k.contiguous(), v.contiguous()  # 3 copies

Each `.contiguous()` launches its own aten copy kernel. This fused kernel
does all three copies in one launch and returns tensors already shaped as
[1, L, num_heads, head_dim] (contiguous).

Registered as `torch.ops.vllm.hcu_rearrange_mixed_qkv` so it is opaque to
Inductor (matching the pattern used by qkv_split_gate_deinterleave).

NOTE(performance): standalone microbenchmark shows this only wins for
L >= ~2048 (long prefill); decode-size L (1..512) is dominated by kernel
launch overhead and the fused version is ~parity with the stock 3-copy
path. Kept in tree so we can flip the qwen3 patch on/off cheaply if the
workload shifts prefill-heavy.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _hcu_split_qkv_kernel(
    MIXED,          # ptr to [L, K + K + V] contiguous input
    Q_OUT,          # ptr to [L, K] contiguous q output
    K_OUT,          # ptr to [L, K] contiguous k output
    V_OUT,          # ptr to [L, V] contiguous v output
    stride_mix,
    stride_q,
    stride_k,
    stride_v,
    L,
    K: tl.constexpr,          # key_dim (num_k * head_k)
    V: tl.constexpr,          # value_dim (num_v * head_v)
    BLOCK_K: tl.constexpr,    # next_pow_of_2(K)
    BLOCK_V: tl.constexpr,    # next_pow_of_2(V)
):
    row = tl.program_id(0)
    if row >= L:
        return

    off_k = tl.arange(0, BLOCK_K)
    mask_k = off_k < K

    # q: MIXED[row, 0:K] -> Q_OUT[row, :]
    q_src = MIXED + row * stride_mix + off_k
    q_dst = Q_OUT + row * stride_q + off_k
    tl.store(q_dst, tl.load(q_src, mask=mask_k, other=0.0), mask=mask_k)

    # k: MIXED[row, K:2K] -> K_OUT[row, :]
    k_src = MIXED + row * stride_mix + K + off_k
    k_dst = K_OUT + row * stride_k + off_k
    tl.store(k_dst, tl.load(k_src, mask=mask_k, other=0.0), mask=mask_k)

    # v: MIXED[row, 2K:2K+V] -> V_OUT[row, :]
    off_v = tl.arange(0, BLOCK_V)
    mask_v = off_v < V
    v_src = MIXED + row * stride_mix + 2 * K + off_v
    v_dst = V_OUT + row * stride_v + off_v
    tl.store(v_dst, tl.load(v_src, mask=mask_v, other=0.0), mask=mask_v)


def hcu_rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused equivalent of Qwen3NextGatedDeltaNet.rearrange_mixed_qkv.

    Returns (q, k, v) shaped [1, L, num_k_heads, head_k_dim],
    [1, L, num_k_heads, head_k_dim], [1, L, num_v_heads, head_v_dim],
    all contiguous. Matches the semantics of the original torch.split +
    rearrange("l (h d) -> 1 l h d") + .contiguous() chain.
    """
    assert mixed_qkv.dim() == 2, (
        f"expected 2D mixed_qkv, got shape={tuple(mixed_qkv.shape)}"
    )
    if not mixed_qkv.is_contiguous():
        mixed_qkv = mixed_qkv.contiguous()

    L, total = mixed_qkv.shape
    K = num_k_heads * head_k_dim
    V = num_v_heads * head_v_dim
    assert total == 2 * K + V, (
        f"mixed_qkv last dim {total} != 2*K+V ({2*K}+{V})"
    )

    q = torch.empty(
        (1, L, num_k_heads, head_k_dim),
        dtype=mixed_qkv.dtype, device=mixed_qkv.device,
    )
    k = torch.empty(
        (1, L, num_k_heads, head_k_dim),
        dtype=mixed_qkv.dtype, device=mixed_qkv.device,
    )
    v = torch.empty(
        (1, L, num_v_heads, head_v_dim),
        dtype=mixed_qkv.dtype, device=mixed_qkv.device,
    )

    # 2D views over the freshly-allocated buffers so the kernel can address
    # them with one row stride. Zero-copy view (they are contiguous).
    q2d = q.view(L, K)
    k2d = k.view(L, K)
    v2d = v.view(L, V)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_V = triton.next_power_of_2(V)

    grid = (L,)
    _hcu_split_qkv_kernel[grid](
        mixed_qkv,
        q2d, k2d, v2d,
        mixed_qkv.stride(0),
        q2d.stride(0),
        k2d.stride(0),
        v2d.stride(0),
        L,
        K=K,
        V=V,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
    )
    return q, k, v


def _hcu_rearrange_mixed_qkv_fake(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = mixed_qkv.shape[0]
    return (
        mixed_qkv.new_empty((1, L, num_k_heads, head_k_dim)),
        mixed_qkv.new_empty((1, L, num_k_heads, head_k_dim)),
        mixed_qkv.new_empty((1, L, num_v_heads, head_v_dim)),
    )


direct_register_custom_op(
    op_name="hcu_rearrange_mixed_qkv",
    op_func=hcu_rearrange_mixed_qkv,
    mutates_args=[],
    fake_impl=_hcu_rearrange_mixed_qkv_fake,
)
