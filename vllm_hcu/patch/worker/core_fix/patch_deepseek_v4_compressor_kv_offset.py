# SPDX-License-Identifier: Apache-2.0
"""Preserve 64-bit offsets in DeepSeek V4 sparse KV cache writes."""
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
PATCH_ID = "worker.core_fix.deepseek_v4.compressor_kv_offset"
_KERNEL = "_finalize_norm_rope_quant_store_sparse_attn"
TARGET_SYMBOL = f"{TARGET_MODULE}.{_KERNEL}"


def apply_to_module(module: ModuleType) -> bool:
    import triton
    from triton.runtime.jit import JITFunction

    target = load_exact_module(TARGET_MODULE, module)
    kernel = getattr(target, _KERNEL, None)
    if not isinstance(kernel, JITFunction) or "k_cache_ptr" not in kernel.arg_names:
        raise PatchCompatibilityError(f"incompatible Triton kernel: {TARGET_SYMBOL}")

    index = kernel.arg_names.index("k_cache_ptr")
    if kernel.params[index].do_not_specialize:
        return False

    # DTK Triton can attach tt.pointer_range=32 to the cache argument.
    # A strided cache can address bytes beyond 2 GiB. Keep this pointer
    # generic so lowering retains the full block offset in every store.
    replacement = triton.jit(
        kernel.fn,
        do_not_specialize=[*kernel.do_not_specialize, "k_cache_ptr"],
        do_not_specialize_on_alignment=kernel.do_not_specialize_on_alignment,
        debug=kernel.debug,
        noinline=kernel.noinline,
    )
    setattr(target, _KERNEL, replacement)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
