# SPDX-License-Identifier: Apache-2.0
"""Bounded capacity slack for dynamically padded HT MoE scratch buffers."""
from math import prod

import torch


def get_moe_workspaces(manager, shapes_and_dtypes, growth_bytes: int):
    """Return exact views while rounding the underlying shared capacity.

    Use the manager's public simultaneous allocation API so its alignment,
    locking, stream and ubatch lifetime rules remain authoritative. A trailing
    byte view reserves slack; it never changes offsets of the real buffers.
    Per slot, additional requested capacity is less than one growth bucket.
    No tensor or manager is cached by this helper.
    """
    if growth_bytes < 0 or growth_bytes % 256:
        raise ValueError("MoE workspace growth must be nonnegative and 256-byte aligned")
    if growth_bytes == 0:
        return manager.get_simultaneous(*shapes_and_dtypes)
    total_bytes = sum(
        ((prod(shape) * dtype.itemsize + 255) // 256) * 256
        for shape, dtype in shapes_and_dtypes
    )
    slack = (-total_bytes) % growth_bytes
    if slack == 0:
        return manager.get_simultaneous(*shapes_and_dtypes)
    views = manager.get_simultaneous(
        *shapes_and_dtypes, ((slack,), torch.uint8)
    )
    return views[:-1]
