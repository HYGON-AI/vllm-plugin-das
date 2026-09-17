# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skip per-token slot mapping for KV cache groups that do not use it.

Backports the ``SlotMappingMode`` behaviour of upstream vLLM (0.26) to the
paired v0.25.1 tree.  Kimi-K3 allocates five KV cache groups: one MLA group
plus four Mamba/GDN groups that keep recurrent state and address the block
table through ``state_indices`` rather than ``slot_mapping``.  Upstream marks
those groups ``SlotMappingMode.NONE`` and returns before launching the kernel.
v0.25.1 has no such concept, so ``MultiGroupBlockTable.compute_slot_mapping``
launches ``_compute_slot_mapping_kernel`` once per group -- five launches per
decode step instead of one.

The signal cannot be threaded through ``BlockTable.__init__`` /
``MultiGroupBlockTable.__init__`` because those classes live in the paired
vLLM tree, which is not editable.  Instead the runner reports the groups that
do not use slot mapping with :func:`set_slot_mapping_groups_without_slots`
after constructing ``InputBatch``, and this module wraps
``MultiGroupBlockTable.compute_slot_mapping`` to skip them.

Fail-open by construction: without an explicit report the stock method runs,
and indices the runner did not name always run.  The only possible effect is
running a kernel upstream would have skipped, never skipping a needed one.
"""

from __future__ import annotations

from types import ModuleType
from typing import Iterable

from vllm.logger import init_logger

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.v1.worker.block_table"
PATCH_ID = "worker.framework_opt.block_table.slot_mapping_modes"
TARGETS = (f"{TARGET_MODULE}.MultiGroupBlockTable.compute_slot_mapping",)

_MARKER = "_vllm_hcu_slot_mapping_modes_applied"
_WRAPPER = "_vllm_hcu_slot_mapping_wrapper"
_SKIP_ATTR = "_hcu_slot_mapping_skip_gids"


def set_slot_mapping_groups_without_slots(
    block_table, group_indices: Iterable[int]
) -> None:
    """Report KV cache groups whose block table is not addressed per token.

    ``group_indices`` positions index the same group list that
    ``MultiGroupBlockTable`` was built from.
    """
    table_count = len(getattr(block_table, "block_tables", ()))
    indices = frozenset(int(index) for index in group_indices)
    out_of_range = [index for index in indices if index < 0 or index >= table_count]
    if out_of_range:
        raise ValueError(
            f"slot mapping group indices {sorted(out_of_range)} are outside the "
            f"KV cache group range [0, {table_count})"
        )
    setattr(block_table, _SKIP_ATTR, indices)


def _hcu_compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
    skip = getattr(self, _SKIP_ATTR, None)
    if not skip:
        return _ORIGINAL(self, num_reqs, query_start_loc, positions)
    for index, block_table in enumerate(self.block_tables):
        if index in skip:
            continue
        block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)


_ORIGINAL = None


def apply_to_module(module: ModuleType) -> bool:
    global _ORIGINAL
    block_table = load_exact_module(TARGET_MODULE, module)
    target = require_class(block_table, "MultiGroupBlockTable", TARGETS[0])
    wrapped = ((target, "compute_slot_mapping", TARGETS[0], _WRAPPER),)
    if already_applied(block_table, _MARKER, wrapped):
        return False
    original = require_callable(target, "compute_slot_mapping", TARGETS[0])
    if getattr(original, _WRAPPER, False):
        # Already wrapped on this class (the module marker only covers the
        # module object, which callers may recreate between applies).
        return False
    _ORIGINAL = original

    def wrapper(self, num_reqs, query_start_loc, positions) -> None:
        return _hcu_compute_slot_mapping(self, num_reqs, query_start_loc, positions)

    setattr(wrapper, _WRAPPER, True)
    setattr(wrapper, "__name__", original.__name__)
    setattr(wrapper, "__doc__", original.__doc__)
    target.compute_slot_mapping = wrapper
    setattr(block_table, _MARKER, True)
    logger.info(
        "hcu patch %s applied to %s; groups reported by the runner as not "
        "addressed per token will skip the slot-mapping launch",
        PATCH_ID,
        TARGETS[0],
    )
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "set_slot_mapping_groups_without_slots",
]
