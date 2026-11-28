# SPDX-License-Identifier: Apache-2.0
"""FP8 sparse-attention workspace using group-major read pages."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pages(Cache, Used, Key, Value, block_stride: tl.constexpr,
                head_stride: tl.constexpr, token_stride: tl.constexpr,
                arena_blocks: tl.constexpr, page: tl.constexpr, tile: tl.constexpr):
    block = tl.load(Used + tl.program_id(0))
    head = tl.program_id(1)
    token = tl.program_id(2) * tile + tl.arange(0, tile)
    column = tl.arange(0, 256)
    mask = (token[:, None] < page) & (column[None, :] < 192)
    src = block * block_stride + head * head_stride + token[:, None] * token_stride + column[None, :]
    dst = ((head * arena_blocks + block) * page + token[:, None]) * 192 + column[None, :]
    k = tl.load(Cache + src, mask, other=0)
    v = tl.load(Cache + src + 192, mask, other=0)
    tl.store(Key + dst, k, mask)
    tl.store(Value + dst, v, mask)


def copy(base, used, workspace, tile=8):
    blocks, heads, page, _ = base.shape
    if used.numel():
        _copy_pages[(used.numel(), heads, triton.cdiv(page, tile))](
            base.view(torch.uint8), used, *workspace, *base.stride()[:3], blocks, page, tile,
            num_warps=4,
        )
    return tuple(t.view(heads, blocks * page, 192).transpose(0, 1).view(base.dtype)
                 for t in workspace)
