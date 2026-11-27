# SPDX-License-Identifier: Apache-2.0
"""Keep the HCU Triton PCP compressor's two predecessor pointers compatible."""


def install():
    from vllm.models.deepseek_v41.common.ops import fused_compress_quant_cache as ops

    kernel = ops._fused_save_compress_norm_kernel
    pointers = ("state", "prev_rows")
    if (not isinstance(kernel, ops.triton.runtime.JITFunction)
            or any(name not in kernel.arg_names for name in pointers)):
        raise RuntimeError("DSV4.1 PCP compressor has an incompatible Triton signature")
    unspecialized = tuple(
        name for name in pointers
        if not kernel.params[kernel.arg_names.index(name)].do_not_specialize
    )
    if not unspecialized:
        return False
    # A predecessor can come from the state ring or from a gathered PCP row.
    # Mixed pointer specialization fails HCU's branch canonicalization.
    ops._fused_save_compress_norm_kernel = ops.triton.jit(
        kernel.fn,
        version=kernel.version,
        do_not_specialize=(*kernel.do_not_specialize, *unspecialized),
        do_not_specialize_on_alignment=kernel.do_not_specialize_on_alignment,
        debug=kernel.debug,
        noinline=kernel.noinline,
        launch_metadata=kernel.launch_metadata,
    )
    return True
