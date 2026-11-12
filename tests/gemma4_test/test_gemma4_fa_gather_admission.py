# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""The NHD gather's loop stride must never truncate to zero.

``write_one_block_bshd`` advances by ``WARP_SIZE / thread_per_dim * ...`` with
integer division. Once an element is wide enough that thread_per_dim exceeds
the 64-wide wave, that stride is zero and the kernel spins on any non-empty
page -- a hung worker rather than a wrong answer. This model's pool
interleaves K and V, so the gather's row width is 2 * 512 = 1024, and BF16 KV
(two bytes, vec_size 8, thread_per_dim 128) lands exactly there while FP8
(one byte, vec_size 16, thread_per_dim 64) does not.
"""

import pytest

from vllm_hcu.models.gemma4 import (
    _NATIVE_WARP_SIZE,
    _gemma4_bshd_stride_is_zero,
)

ROW_WIDTH = 2 * 512


@pytest.mark.parametrize(
    "itemsize, expected",
    [
        (1, False),   # FP8: vec_size 16, thread_per_dim 64 == wave size
        (2, True),    # BF16/FP16: vec_size 8, thread_per_dim 128
        (4, True),    # FP32: vec_size 4, thread_per_dim 256
    ],
)
def test_interleaved_row_width_by_element_size(itemsize, expected):
    assert _gemma4_bshd_stride_is_zero(ROW_WIDTH, itemsize) is expected


def test_the_boundary_is_the_wave_size():
    """thread_per_dim may equal the wave size, but not exceed it."""
    assert _gemma4_bshd_stride_is_zero(_NATIVE_WARP_SIZE * 16, 1) is False
    assert _gemma4_bshd_stride_is_zero(_NATIVE_WARP_SIZE * 16 + 16, 1) is True


def test_a_single_head_width_is_fine_in_bf16():
    """The bound is about the row width, not about BF16 as such."""
    assert _gemma4_bshd_stride_is_zero(512, 2) is False


def test_an_element_wider_than_the_vector_is_rejected():
    assert _gemma4_bshd_stride_is_zero(ROW_WIDTH, 32) is True
