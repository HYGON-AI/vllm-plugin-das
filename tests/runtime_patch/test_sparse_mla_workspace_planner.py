# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm_hcu.v1.attention.backends.mla.prefill_workspace import plan_prefill_workspace


def _check_plan(table, lengths, block_size, capacity):
    chunks = plan_prefill_workspace(table, lengths, block_size, capacity)
    next_row = 0
    for chunk in chunks:
        assert chunk.req_start == next_row
        next_row = chunk.req_end
        assert 0 <= chunk.num_tokens <= capacity
        assert len(set(chunk.physical_blocks.tolist())) == len(chunk.physical_blocks)
        assert chunk.valid_tokens.sum() == chunk.num_tokens
        sources = []
        for physical, valid, start in zip(
            chunk.physical_blocks, chunk.valid_tokens, chunk.workspace_starts
        ):
            assert start == len(sources)
            sources.extend((physical, offset) for offset in range(valid))
        for local_row, row in enumerate(range(chunk.req_start, chunk.req_end)):
            for token in range(lengths[row]):
                page, offset = divmod(token, block_size)
                workspace_row = chunk.row_block_starts[local_row, page] + offset
                assert sources[workspace_row] == (table[row, page], offset)
    assert next_row == len(lengths)
    return chunks


def test_shared_prefix_and_pcp_segments():
    # Rows 0 and 2 are two PCP segments of one request. Row 1 is another
    # request sharing the first two pages.
    table = np.array([[5, 1, 7], [5, 1, 8], [5, 1, 7]], dtype=np.int32)
    chunks = _check_plan(table, [130, 150, 190], 64, 256)
    assert len(chunks) == 1
    assert chunks[0].num_tokens == 64 + 64 + 62 + 22
    np.testing.assert_array_equal(chunks[0].physical_blocks, [5, 1, 7, 8])
    np.testing.assert_array_equal(
        chunks[0].row_block_starts[0], chunks[0].row_block_starts[2]
    )


def test_partial_page_extension_moves_later_pages():
    table = np.array([[3, 7], [3, 9], [3, 7]], dtype=np.int32)
    chunks = _check_plan(table, [65, 70, 127], 64, 133)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0].valid_tokens, [64, 63, 6])
    np.testing.assert_array_equal(chunks[0].workspace_starts, [0, 64, 127])


def test_partial_extension_splits_group_without_committing_candidate():
    table = np.array([[3, 7], [3, 9], [3, 7]], dtype=np.int32)
    chunks = _check_plan(table, [65, 70, 127], 64, 128)
    assert [(c.req_start, c.req_end, c.num_tokens) for c in chunks] == [
        (0, 2, 71),
        (2, 3, 127),
    ]
    np.testing.assert_array_equal(chunks[0].valid_tokens, [64, 1, 6])
    np.testing.assert_array_equal(chunks[1].workspace_starts, [0, 64])


def test_unique_pages_and_exact_capacity():
    table = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    chunks = _check_plan(table, [100, 128, 1], 64, 228)
    assert [(c.req_start, c.req_end) for c in chunks] == [(0, 2), (2, 3)]


def test_repeated_physical_page_within_one_row():
    table = np.array([[2, 2]], dtype=np.int32)
    chunks = _check_plan(table, [100], 64, 100)
    assert chunks[0].num_tokens == 64
    np.testing.assert_array_equal(chunks[0].row_block_starts, [[0, 0]])


def test_empty_rows_and_unused_block_table_entries():
    table = np.array([[99, -1], [2, -1], [99, -1]], dtype=np.int32)
    chunks = _check_plan(table, [0, 10, 0], 64, 64)
    assert chunks[0].physical_blocks.tolist() == [2]
    assert plan_prefill_workspace(np.empty((0, 0), dtype=np.int32), [], 64, 64) == []


@pytest.mark.parametrize("seed", range(20))
def test_random_page_aliasing(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        block_size = int(rng.choice([1, 8, 16, 64, 128]))
        capacity = block_size * 5 + 3
        lengths = rng.integers(0, capacity + 1, size=12).tolist()
        max_blocks = (capacity + block_size - 1) // block_size
        table = rng.integers(0, 14, size=(12, max_blocks), dtype=np.int32)
        _check_plan(table, lengths, block_size, capacity)


@pytest.mark.parametrize(
    "table,lengths,block_size,capacity",
    [
        ([[1]], [65], 64, 64),
        ([[1]], [-1], 64, 64),
        ([[1]], [65], 64, 128),
        ([[-1]], [1], 64, 64),
        ([[1]], [], 64, 64),
        ([[1]], [1], 0, 64),
        ([[1]], [1], 64, 0),
    ],
)
def test_invalid_inputs(table, lengths, block_size, capacity):
    with pytest.raises(ValueError):
        plan_prefill_workspace(np.asarray(table), lengths, block_size, capacity)
