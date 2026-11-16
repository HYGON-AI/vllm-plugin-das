# SPDX-License-Identifier: Apache-2.0
from types import ModuleType

from ._common import PatchCompatibilityError


def preserve_cache_pointer_range(module: ModuleType, symbol: str) -> bool:
    import triton
    from triton.runtime.jit import JITFunction

    kernel = getattr(module, symbol, None)
    if not isinstance(kernel, JITFunction) or "k_cache_ptr" not in kernel.arg_names:
        raise PatchCompatibilityError(f"incompatible Triton kernel: {module.__name__}.{symbol}")

    index = kernel.arg_names.index("k_cache_ptr")
    if kernel.params[index].do_not_specialize:
        return False

    # DTK Triton can attach tt.pointer_range=32 to the cache argument.
    # A strided cache can address bytes beyond 2 GiB. Keep this pointer
    # generic so lowering retains the full block offset for cache accesses.
    replacement = triton.jit(
        kernel.fn,
        do_not_specialize=[*kernel.do_not_specialize, "k_cache_ptr"],
        do_not_specialize_on_alignment=kernel.do_not_specialize_on_alignment,
        debug=kernel.debug,
        noinline=kernel.noinline,
    )
    setattr(module, symbol, replacement)
    return True
