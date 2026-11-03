# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PrefillWorkspaceChunk:
    req_start: int
    req_end: int
    physical_blocks: np.ndarray
    valid_tokens: np.ndarray
    workspace_starts: np.ndarray
    row_block_starts: np.ndarray
    num_tokens: int


def plan_prefill_workspace(
    block_table: np.ndarray,
    seq_lens: Sequence[int],
    block_size: int,
    workspace_size: int,
) -> list[PrefillWorkspaceChunk]:
    """Pack consecutive prefill rows using their unique physical KV pages.

    Partial pages contribute the largest extent referenced by any row in a
    chunk. Page identities are local to the supplied cache group and rank.
    """
    if block_size <= 0 or not 0 < workspace_size <= np.iinfo(np.int32).max:
        raise ValueError("Invalid block or workspace size")
    if block_table.ndim != 2 or len(seq_lens) != len(block_table):
        raise ValueError("Block table and sequence lengths must have matching rows")

    lengths = np.asarray(seq_lens, dtype=np.int64)
    if np.any(lengths < 0) or np.any(lengths > workspace_size):
        raise ValueError("Each sequence must fit in the workspace")
    num_blocks = (lengths + block_size - 1) // block_size
    if np.any(num_blocks > block_table.shape[1]):
        raise ValueError("Block table does not cover the sequence")
    if len(lengths) == 0:
        return []
    references = np.concatenate([row[:n] for row, n in zip(block_table, num_blocks)])
    if not np.issubdtype(references.dtype, np.integer) or np.any(references < 0):
        raise ValueError("Physical block IDs must be nonnegative integers")
    if np.any(references > np.iinfo(np.int32).max):
        raise ValueError("Physical block IDs must fit in int32")

    # Compress page identities once for the whole batch. Array updates below
    # avoid a Python dictionary lookup for every page reference of every row.
    physical, first, inverse = np.unique(
        references, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    physical = physical[order].astype(np.int32)
    remap = np.empty(len(order), dtype=np.intp)
    remap[order] = np.arange(len(order))
    inverse = remap[inverse]
    boundaries = np.concatenate(([0], np.cumsum(num_blocks)))
    row_ids = [inverse[boundaries[i] : boundaries[i + 1]] for i in range(len(lengths))]
    extents = np.zeros(len(physical), dtype=np.int32)
    chunks: list[PrefillWorkspaceChunk] = []
    start, num_tokens = 0, 0

    def finish(end: int) -> None:
        active = np.flatnonzero(extents)
        valid_tokens = extents[active]
        starts = np.zeros(len(active), dtype=np.int32)
        if len(active) > 1:
            np.cumsum(valid_tokens[:-1], out=starts[1:])
        offsets = np.full(len(physical), -1, dtype=np.int32)
        offsets[active] = starts
        max_blocks = int(num_blocks[start:end].max(initial=0))
        row_starts = np.full((end - start, max_blocks), -1, dtype=np.int32)
        for i, ids in enumerate(row_ids[start:end]):
            row_starts[i, : len(ids)] = offsets[ids]
        chunks.append(
            PrefillWorkspaceChunk(
                start,
                end,
                physical[active],
                valid_tokens,
                starts,
                row_starts,
                num_tokens,
            )
        )

    for i, ids in enumerate(row_ids):
        valid = np.full(len(ids), block_size, dtype=np.int32)
        if len(valid):
            valid[-1] = (int(lengths[i]) - 1) % block_size + 1
        # Repeated page IDs within a row must not count its extent twice.
        unique, positions = np.unique(ids, return_inverse=True)
        required = np.zeros(len(unique), dtype=np.int32)
        np.maximum.at(required, positions, valid)
        delta = int(np.maximum(required - extents[unique], 0).sum())
        if num_tokens + delta > workspace_size:
            finish(i)
            start, num_tokens = i, 0
            extents.fill(0)
            delta = int(required.sum())
        extents[unique] = np.maximum(extents[unique], required)
        num_tokens += delta
    finish(len(lengths))
    return chunks
