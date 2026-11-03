# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np


class BlockTableMirror:
    """Optional host copy of canonical kernel-page IDs for prefill planning."""

    def __init__(self, num_reqs: int, num_blocks: int):
        self.table = np.zeros((num_reqs, num_blocks), dtype=np.int32)
        self.lengths = np.zeros(num_reqs, dtype=np.int32)

    def update(self, req_index: int, block_ids: list[int], overwrite: bool) -> None:
        start = 0 if overwrite else int(self.lengths[req_index])
        end = start + len(block_ids)
        if end > self.table.shape[1]:
            raise ValueError("Block IDs exceed the host block table capacity")
        if overwrite:
            self.table[req_index].fill(0)
        self.table[req_index, start:end] = block_ids
        self.lengths[req_index] = end

    def gather(self, req_indices: np.ndarray, num_reqs_padded: int) -> np.ndarray:
        if len(req_indices) > num_reqs_padded:
            raise ValueError("Request indices exceed the padded batch size")
        result = np.zeros((num_reqs_padded, self.table.shape[1]), dtype=np.int32)
        result[: len(req_indices)] = self.table[req_indices]
        return result


HOST_TABLE_ATTRIBUTE = "_hcu_sparse_mla_workspace_cpu_table"


def install_workspace_block_table_mirrors(block_tables) -> None:
    """Mirror host append operations before the first request is admitted."""
    import functools

    if hasattr(block_tables, "_hcu_workspace_mirrors"):
        return
    if np.any(block_tables.num_blocks.np):
        raise RuntimeError(
            "Workspace mirrors must be installed before block allocation"
        )
    mirrors = [
        BlockTableMirror(*table.gpu.shape) for table in block_tables.block_tables
    ]
    original = block_tables.append_block_ids

    @functools.wraps(original)
    def append(req_index, new_block_ids, overwrite):
        original(req_index, new_block_ids, overwrite)
        for group, mirror in enumerate(mirrors):
            factor = block_tables.blocks_per_kv_block[group]
            pages = [
                block * factor + offset
                for block in new_block_ids[group]
                for offset in range(factor)
            ]
            mirror.update(req_index, pages, overwrite)

    block_tables.append_block_ids = append
    block_tables._hcu_workspace_mirrors = mirrors


def attach_workspace_block_tables(block_tables, tables, input_batch):
    """Attach rank-local snapshots in the device tables' current row order."""
    mirrors = getattr(block_tables, "_hcu_workspace_mirrors", None)
    if mirrors is None or not np.any(input_batch.is_prefilling_np):
        return tables
    if len(mirrors) != len(tables):
        raise ValueError("Workspace host and device cache groups do not match")
    for mirror, table in zip(mirrors, tables):
        host = mirror.gather(input_batch.idx_mapping_np, table.shape[0])
        setattr(table, HOST_TABLE_ATTRIBUTE, host)
    return tables
