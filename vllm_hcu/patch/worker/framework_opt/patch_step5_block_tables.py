# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.v1.worker.gpu.block_table"
PATCH_ID = "worker.framework_opt.step5_block_tables"
TARGETS = ("vllm.v1.worker.gpu.block_table.BlockTables.append_block_ids",)
_MARKER = "_vllm_hcu_step5_block_tables_applied"

SIGNATURES = (
    (
        "vllm.v1.worker.gpu.block_table.BlockTables.append_block_ids",
        ("self", "req_index", "new_block_ids", "overwrite"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os
    import numpy as np

    cls = getattr(target, "BlockTables", None)
    original = require_callable(cls, "append_block_ids", TARGETS[0])

    def enable_host_mirror(self):
        if getattr(self, "_vllm_hcu_block_tables_np", None) is None:
            if self.num_blocks.np.any():
                raise RuntimeError(
                    "Enable the host block mirror before adding requests"
                )
            self._vllm_hcu_block_tables_np = [
                np.zeros(tuple(b.gpu.shape), dtype=np.int32) for b in self.block_tables
            ]
            self._vllm_hcu_compact_block_tables = (
                os.environ.get("HCU_STEP5_COMPACT_HOST_BLOCK_TABLES", "0") == "1"
            )
            if self._vllm_hcu_compact_block_tables:
                print(
                    "[step5_host_tables] compact host snapshots enabled; "
                    "full context backing retained",
                    flush=True,
                )

    def gather_block_tables_np(self, idx_mapping_np, num_reqs_padded):
        tables = getattr(self, "_vllm_hcu_block_tables_np", None)
        if tables is None:
            raise RuntimeError("The HCU host block mirror is not enabled")
        out = []
        for group, table in enumerate(tables):
            width = table.shape[1]
            if self._vllm_hcu_compact_block_tables:
                # Counts already include newly scheduled pages and kernel-block
                # expansion. Keep the full backing mirror for 1M contexts; only
                # compact this batch's independent host snapshot. Padded rows
                # may read column zero even when no request has any pages.
                counts = self.num_blocks.np[group, idx_mapping_np]
                width = min(width, max(1, int(counts.max(initial=0))))
                gathered = np.zeros((num_reqs_padded, width), dtype=table.dtype)
                # np.take on a sliced noncontiguous source can first copy all
                # request rows. Advanced indexing copies only selected rows.
                gathered[: len(idx_mapping_np)] = table[idx_mapping_np, :width]
            else:
                gathered = np.zeros((num_reqs_padded, width), dtype=table.dtype)
                np.take(table, idx_mapping_np, axis=0, out=gathered[: len(idx_mapping_np)])
            out.append(gathered)
        return tuple(out)

    @functools.wraps(original)
    def append_block_ids(self, req_index, new_block_ids, overwrite):
        result = original(self, req_index, new_block_ids, overwrite)
        tables = getattr(self, "_vllm_hcu_block_tables_np", None)
        if tables is not None:
            for i, ids in enumerate(new_block_ids):
                bpk = self.blocks_per_kv_block[i]
                expanded = (
                    [b * bpk + k for b in ids for k in range(bpk)] if bpk > 1 else ids
                )
                end = self.num_blocks.np[i, req_index]
                tables[i][req_index, end - len(expanded) : end] = expanded
        return result

    cls.enable_host_mirror = enable_host_mirror
    cls.gather_block_tables_np = gather_block_tables_np
    cls.append_block_ids = append_block_ids
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
