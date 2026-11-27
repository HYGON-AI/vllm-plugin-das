# SPDX-License-Identifier: Apache-2.0
"""Keep large SWA cache offsets 64-bit in the DSV4.1 ROCm gather."""

_CACHE_POINTER = 'k_cache_ptr'


def install():
    from vllm.models.deepseek_v41.common.ops import cache_utils

    kernel = cache_utils._dequantize_and_gather_k_one_pass_kernel
    triton = cache_utils.triton
    if (not isinstance(kernel, triton.runtime.JITFunction)
            or _CACHE_POINTER not in kernel.arg_names):
        raise RuntimeError('DSV4.1 ROCm KV gather has an incompatible Triton signature')

    pointer_index = kernel.arg_names.index(_CACHE_POINTER)
    if kernel.params[pointer_index].do_not_specialize:
        return False

    # ROCm can narrow the explicit int64 page offset to int32 when the
    # cache pointer is specialized for buffer loads. Only opt out for
    # this pointer so unrelated kernels keep their normal compilation.
    cache_utils._dequantize_and_gather_k_one_pass_kernel = triton.jit(
        kernel.fn,
        version=kernel.version,
        do_not_specialize=(*kernel.do_not_specialize, _CACHE_POINTER),
        do_not_specialize_on_alignment=kernel.do_not_specialize_on_alignment,
        debug=kernel.debug,
        noinline=kernel.noinline,
        launch_metadata=kernel.launch_metadata,
    )
    return True
