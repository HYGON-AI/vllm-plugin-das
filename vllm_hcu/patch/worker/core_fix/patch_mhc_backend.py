# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Make the target vLLM mHC backend honor HCU runtime capabilities."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_backend_switch"
_MARKER = "_vllm_hcu_mhc_backend_switch_applied"


def _tilelang_runtime_available() -> bool:
    """Require a usable TileLang runtime, not merely an installed package."""

    try:
        import tilelang  # noqa: F401
    except Exception:
        return False
    return True


def _aiter_mhc_runtime_complete(mhc) -> bool:
    """Require the HCU delayed adapter and its actual staged AITER operators."""

    try:
        from vllm._aiter_ops import rocm_aiter_ops

        from vllm_hcu.model_executor.layers import mhc_runtime

        capabilities = mhc_runtime.runtime()
        return (
            mhc_runtime.pre_enabled()
            and capabilities.delayed_available
            and (capabilities.post is not None
                 or capabilities.tilelang_post is not None)
            and all(
                callable(getattr(rocm_aiter_ops, name, None))
                for name in (
                    "mhc_pre_delayed",
                    "mhc_fused_post_pre_delayed_prefers_unfused",
                )
            )
        )
    except Exception:
        return False


def _aiter_mhc_post_runtime_available() -> bool:
    """Resolve Post independently of optional Pre and TileLang dependencies."""
    from vllm_hcu.model_executor.layers import mhc_runtime

    ops = mhc_runtime.runtime()
    return ops.post is not None or ops.tilelang_post is not None


def _aiter_mhc_post_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> bool:
    if residual.ndim < 3:
        return False
    if residual.dtype == torch.float16:
        from vllm_hcu.model_executor.layers import mhc_runtime

        if mhc_runtime.runtime().post is None:
            return False
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    return (
        residual.numel() > 0
        and residual.shape[-2] == 4
        and hidden_size >= 1024
        # The installed v3 b32/b64 variants fail numerical probes; use b128 only.
        and hidden_size % 1024 == 0
        and x.shape == (*outer_shape, hidden_size)
        and post_layer_mix.shape
        in (
            (*outer_shape, 4),
            (*outer_shape, 4, 1),
        )
        and comb_res_mix.shape == (*outer_shape, 4, 4)
        and residual.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == residual.dtype
        and post_layer_mix.dtype == comb_res_mix.dtype == torch.float32
        and all(
            tensor.is_cuda
            and tensor.device == residual.device
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


def _aiter_mhc_post_enabled() -> bool:
    import vllm_hcu.platforms.envs as henvs

    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_USE_AITER_MHC and henvs.VLLM_HCU_USE_AITER_MHC_POST
    )


def _call_aiter_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.mhc_post_aiter(x, residual, post_layer_mix, comb_res_mix)


def _prepare_mhc_post_patch(
    mhc: ModuleType,
) -> tuple[type, Callable[..., torch.Tensor], Callable[..., torch.Tensor], bool]:
    post_op = require_class(mhc, "MHCPostOp", f"{TARGET_MODULE}.MHCPostOp")
    original = require_callable(
        post_op, "forward_hip", f"{TARGET_MODULE}.MHCPostOp.forward_hip"
    )
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.MHCPostOp.forward_hip",
        positional=("self", "x", "residual", "post_layer_mix", "comb_res_mix"),
    )
    require_callable(
        post_op, "forward_native", f"{TARGET_MODULE}.MHCPostOp.forward_native"
    )
    post_available = _aiter_mhc_post_runtime_available()

    @functools.wraps(original)
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if not _aiter_mhc_post_enabled():
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)
        if post_available:
            if _aiter_mhc_post_supported(x, residual, post_layer_mix, comb_res_mix):
                return _call_aiter_mhc_post(x, residual, post_layer_mix, comb_res_mix)
            # Empty, strided and unsupported layouts must not reach the HIP kernel.
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)
        return original(self, x, residual, post_layer_mix, comb_res_mix)

    return post_op, original, forward_hip, post_available


def apply_to_module(module: ModuleType) -> bool:
    mhc = load_exact_module(TARGET_MODULE, module)
    if getattr(mhc, _MARKER, False):
        if (
            not hasattr(mhc, "_vllm_hcu_original_has_aiter_mhc")
            or not hasattr(mhc, "_vllm_hcu_original_has_tilelang_mhc")
            or not hasattr(mhc, "_vllm_hcu_original_post_forward_hip")
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    has_aiter_mhc = getattr(mhc, "HAS_AITER_MHC", None)
    if not isinstance(has_aiter_mhc, bool):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.HAS_AITER_MHC is missing"
        )
    has_tilelang_mhc = getattr(mhc, "HAS_TILELANG_MHC", None)
    if not isinstance(has_tilelang_mhc, bool):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.HAS_TILELANG_MHC is missing"
        )

    import vllm_hcu.platforms.envs as henvs

    post_op, original_post, post_forward, post_available = _prepare_mhc_post_patch(mhc)

    # Preserve target capability detection while honoring the HCU AITER switch.
    # Upstream's TileLang probe only checks package discovery; the DTK image can
    # contain an installed TileLang wheel whose tvm_ffi ABI is incompatible.
    # Probe the real import once so official forward_hip methods safely fall
    # back to their native Torch/Triton implementations instead of failing on
    # the first profile run.
    mhc._vllm_hcu_original_has_aiter_mhc = has_aiter_mhc
    mhc._vllm_hcu_original_has_tilelang_mhc = has_tilelang_mhc
    mhc.HAS_AITER_MHC = bool(
        has_aiter_mhc
        and henvs.optional_custom_op_enabled(henvs.VLLM_HCU_USE_AITER_MHC)
        and _aiter_mhc_runtime_complete(mhc)
    )
    mhc.HAS_TILELANG_MHC = bool(has_tilelang_mhc and _tilelang_runtime_available())
    mhc._vllm_hcu_original_post_forward_hip = original_post
    post_op.forward_hip = post_forward
    if post_available and _aiter_mhc_post_enabled():
        # Resolve once before model warmup; never change the kernel during capture.
        os.environ.setdefault("AITER_MHC_POST_KERNEL", "v3")
    setattr(mhc, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
