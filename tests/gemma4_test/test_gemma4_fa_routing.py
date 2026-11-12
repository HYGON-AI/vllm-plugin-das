# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""_can_use_fa must decline the KV dtypes the native gather cannot stride.

The arithmetic itself is covered in test_gemma4_fa_gather_admission.py. This
file checks the wiring: that the admission reads the resolved layout and the
pool's element size, so a BF16 pool under --kv-cache-dtype auto routes to
Triton instead of reaching a loop whose stride truncates to zero.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.gemma4 import Gemma4FAAttentionImpl

HEAD_SIZE = 512
NUM_HEADS = 16
NUM_KV_HEADS = 2
BLOCK_SIZE = 64
ROW_WIDTH = 2 * HEAD_SIZE


def kv_cache(dtype, layout):
    """A pool whose strides name one of the two native selectors."""
    if layout == "NHD":
        # Contiguous as [blocks, block_size, kv_heads, row_width]; the view
        # helper transposes it back to the logical order.
        pool = torch.zeros(
            (4, BLOCK_SIZE, NUM_KV_HEADS, ROW_WIDTH), dtype=dtype, device="cuda"
        )
        return pool.transpose(1, 2)
    return torch.zeros(
        (4, NUM_KV_HEADS, BLOCK_SIZE, ROW_WIDTH), dtype=dtype, device="cuda"
    )


def impl(kv_quant_mode):
    obj = object.__new__(Gemma4FAAttentionImpl)
    obj.head_size = HEAD_SIZE
    obj.num_heads = NUM_HEADS
    obj.num_kv_heads = NUM_KV_HEADS
    obj._kv_quant_mode = kv_quant_mode
    obj._fa_static_failures = ()
    obj._log_fa_failures = lambda *a, **k: None
    return obj


def metadata():
    return SimpleNamespace(
        max_query_len=512,
        num_actual_tokens=4096,
        use_cascade=False,
        causal=True,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
    )


def decide(dtype, layout, kv_quant_mode):
    query = torch.zeros(
        (4096, NUM_HEADS, HEAD_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    return impl(kv_quant_mode)._can_use_fa(
        query, kv_cache(dtype, layout), metadata(), query, None, None, "test"
    )


@pytest.mark.hcu
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_fp8_pool_is_admitted(layout):
    from vllm.v1.kv_cache_interface import KVQuantMode

    assert decide(torch.float8_e4m3fn, layout, KVQuantMode.FP8_PER_TENSOR) is True


@pytest.mark.hcu
def test_bf16_nhd_pool_is_declined():
    """The combination --kv-cache-dtype auto produces, and the one that hangs."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    assert decide(torch.bfloat16, "NHD", KVQuantMode.NONE) is False


@pytest.mark.hcu
def test_bf16_hnd_pool_is_unaffected():
    """HND uses write_one_block_bhsd, whose stride cannot reach zero."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    assert decide(torch.bfloat16, "HND", KVQuantMode.NONE) is True
