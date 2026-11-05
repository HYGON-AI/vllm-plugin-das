# SPDX-License-Identifier: Apache-2.0
"""Step5 W4A8 operator provided by the normal vllm_hcu.hcu_ops build."""
from functools import lru_cache
import importlib


@lru_cache(maxsize=1)
def require_native_op():
    import torch

    importlib.import_module("vllm_hcu.hcu_ops")
    try:
        op = torch.ops.hcu_ops.step5_w4a8_gemm.default
    except AttributeError as error:
        raise RuntimeError(
            "The installed HCU plugin has no Step5 W4A8 operator. Rebuild/install "
            "the plugin with VLLM_HCU_BUILD_STEP5_W4A8=1 and PYTORCH_ROCM_ARCH=gfx938."
        ) from error
    for dispatch in ("CUDA", "Meta"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "hcu_ops::step5_w4a8_gemm", dispatch
        ):
            raise RuntimeError(f"Step5 W4A8 {dispatch} implementation is missing")
    return op
