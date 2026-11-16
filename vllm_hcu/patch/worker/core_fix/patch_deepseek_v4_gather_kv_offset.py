# SPDX-License-Identifier: Apache-2.0
"""Preserve 64-bit offsets in DeepSeek V4 sparse KV cache reads."""
from types import ModuleType

from ._common import load_exact_module
from ._triton_kv_offset import preserve_cache_pointer_range

TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.cache_utils"
PATCH_ID = "worker.core_fix.deepseek_v4.gather_kv_offset"
_KERNEL = "_dequantize_and_gather_k_kernel"
TARGET_SYMBOL = f"{TARGET_MODULE}.{_KERNEL}"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return preserve_cache_pointer_range(target, _KERNEL, "k_cache_ptr")


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
