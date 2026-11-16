# SPDX-License-Identifier: Apache-2.0
"""Preserve 64-bit offsets in DeepSeek V4 sparse compressor state writes."""
from types import ModuleType

from ._common import load_exact_module
from ._triton_kv_offset import preserve_cache_pointer_range

TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.save_partial_states"
PATCH_ID = "worker.core_fix.deepseek_v4.state_cache_offset"
_KERNEL = "_save_partial_states_kernel"
TARGET_SYMBOL = f"{TARGET_MODULE}.{_KERNEL}"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return preserve_cache_pointer_range(target, _KERNEL, "state_cache_ptr")


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
