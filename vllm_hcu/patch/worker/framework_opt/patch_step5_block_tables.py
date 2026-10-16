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

    def gather_block_tables_np(self, idx_mapping_np, num_reqs_padded):
        tables = getattr(self, "_vllm_hcu_block_tables_np", None)
        if tables is None:
            raise RuntimeError("The HCU host block mirror is not enabled")
        out = []
        for table in tables:
            gathered = np.zeros((num_reqs_padded, table.shape[1]), dtype=table.dtype)
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
