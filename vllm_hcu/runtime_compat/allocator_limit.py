# SPDX-License-Identifier: Apache-2.0
"""Optional allocator limit shared by standard HCU workers; unset means no change."""

import math
import os


def apply_allocator_limit(device, torch_module=None):
    raw = os.environ.get("VLLM_HCU_ALLOCATOR_LIMIT_GIB")
    if raw is None:
        return None
    gib = float(raw)
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("VLLM_HCU_ALLOCATOR_LIMIT_GIB must be finite and positive")
    if torch_module is None:
        import torch as torch_module
    total = torch_module.cuda.get_device_properties(device).total_memory
    cap = int(gib * 2**30)
    torch_module.cuda.set_per_process_memory_fraction(min(cap / total, 0.98), device)
    return cap
