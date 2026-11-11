# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical regressions for BF16 indexer cache gather addressing."""

import pytest
import torch

from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    cp_gather_indexer_k_bf16_cache_triton,
)

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU"),
]


@pytest.mark.parametrize("strided", [False, True])
def test_gather_block_table_strides(strided):
    cache = torch.arange(8 * 64 * 128, device="cuda", dtype=torch.float32)
    cache = cache.to(torch.bfloat16).view(8, 64, 128)
    backing = torch.tensor(
        [[0, 0, 0, 0], [1, 7, 2, 7], [3, 7, 4, 7]],
        device="cuda", dtype=torch.int32,
    )
    table = backing[1:, ::2]
    if not strided:
        table = table.contiguous()
    cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    expected = torch.cat([cache[1], cache[2], cache[3], cache[4]])
    actual = torch.empty_like(expected)
    cp_gather_indexer_k_bf16_cache_triton(cache, actual, table, cu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gather_cache_offset_exceeds_signed_int32():
    # Model layers can share an interleaved allocation: even a small logical
    # cache view can address more than 2**31 elements from its first page.
    page_stride = 2**24
    required_bytes = (129 * page_stride + 64 * 128) * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes + 512 * 1024**2:
        pytest.skip("requires 4.6 GiB free memory for 64-bit address regression")
    cache = torch.empty_strided(
        (130, 64, 128), (page_stride, 128, 1),
        dtype=torch.bfloat16, device="cuda",
    )
    expected = torch.arange(64 * 128, device="cuda", dtype=torch.float32)
    expected = expected.to(torch.bfloat16).view(64, 128)
    cache[129].copy_(expected)
    table = torch.tensor([[129]], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    cp_gather_indexer_k_bf16_cache_triton(cache, actual, table, cu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Replaying the same captured call must also retain 64-bit offsets.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cp_gather_indexer_k_bf16_cache_triton(cache, actual, table, cu)
    actual.zero_()
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
