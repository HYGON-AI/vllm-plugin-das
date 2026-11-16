# SPDX-License-Identifier: Apache-2.0
"""Preserve 64-bit offsets in DeepSeek V4 sparse KV cache writes."""
from types import ModuleType

from ._common import load_exact_module
from ._triton_kv_offset import preserve_cache_pointer_range

TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
PATCH_ID = "worker.core_fix.deepseek_v4.compressor_kv_offset"
_KERNEL = "_finalize_norm_rope_quant_store_sparse_attn"
TARGET_SYMBOL = f"{TARGET_MODULE}.{_KERNEL}"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return preserve_cache_pointer_range(target, _KERNEL)


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
