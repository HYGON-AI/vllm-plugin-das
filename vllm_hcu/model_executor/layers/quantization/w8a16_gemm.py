# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main (0.18) modelopt.py w8a16 stack: deepgemm W8A16
# GEMM with tuned-config layout awareness (marlin pre-shuffle support), a
# one-pass marlin dequant Triton kernel for the large-M path, and a PyTorch
# dequant fallback, registered as torch.ops.vllm.w8a16_gemm (the registration
# point the 0.18 overlay owned).

from __future__ import annotations

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)

# W8A16 weight layouts, matching deepgemm.w8a16_linear's WEIGHT_LAYOUT.
W8A16_LAYOUT_RAW = 0
W8A16_LAYOUT_MARLIN = 1
# Largest M probed at load time when deciding whether DeepGEMM's tuned configs
# want the marlin layout for a given (N, K).
_W8A16_MARLIN_PROBE_MAX_M = 256


def _deepgemm_version() -> str:
    # Package metadata can disagree with the module actually imported (e.g. a
    # source build on PYTHONPATH), so report both.
    try:
        from importlib.metadata import version
        ver = version("deepgemm")
    except Exception:
        ver = "unknown"
    try:
        import deepgemm
        return f"{ver} ({os.path.dirname(deepgemm.__file__)})"
    except ImportError:
        return ver


@functools.cache
def _w8a16_tuned_marlin(n: int, k: int) -> bool:
    """Whether any DeepGEMM tuned config for M <= 256 picks the marlin layout."""
    import deepgemm
    return any(
        int(deepgemm.get_w8a16_linear_config(m, n, k).get("WEIGHT_LAYOUT", 0))
        == W8A16_LAYOUT_MARLIN
        for m in range(1, _W8A16_MARLIN_PROBE_MAX_M + 1)
    )


def w8a16_should_preshuffle(n: int, k: int) -> bool:
    """Whether an [N, K] int8 weight should be stored in the marlin layout."""
    if not (henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_DEEPGEMM_W8A16_GEMM
            and henvs.VLLM_HCU_DEEPGEMM_W8A16_PRESHUFFLE):
        return False
    if n % 16 != 0 or k % 128 != 0:
        return False
    try:
        import deepgemm
    except ImportError:
        return False
    if not hasattr(deepgemm, "w8a16_marlin_weight"):
        return False
    return _w8a16_tuned_marlin(n, k)


def w8a16_unmarlin_weight(w: torch.Tensor) -> torch.Tensor:
    """Inverse of deepgemm.w8a16_marlin_weight: marlin -> row-major [N, K]."""
    n, k = w.shape
    w_u32 = w.contiguous().view(torch.int32).view(k // 128, n // 16, 4, 16, 8)
    w_u32 = w_u32.transpose(0, 1).contiguous().view(n // 16, k // 32, 16, 8)
    return w_u32.transpose(1, 2).contiguous().view(torch.int8).view(n, k)


def _w8a16_dequant(
    w: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    weight_layout: int,
) -> torch.Tensor:
    """Dequantize a W8A16 weight in either layout to row-major [N, K] W16.

    Uncached, so it is safe inside CUDA graph capture.
    """
    if weight_layout == W8A16_LAYOUT_MARLIN:
        from vllm_hcu.model_executor.layers.quantization import (
            w8a16_marlin_utils)
        return w8a16_marlin_utils.w8a16_marlin_dequant(w, scale, dtype)
    import deepgemm
    if dtype == torch.bfloat16:
        return deepgemm.op.w8a16_dequant_weight_bf16(w, scale)
    return deepgemm.op.w8a16_dequant_weight(w, scale)


def _w8a16_gemm_deepgemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    weight_layout: int,
) -> torch.Tensor:
    """DeepGEMM W8A16 GEMM with the weight's actual storage layout.

    DeepGEMM's tuned configs choose WEIGHT_LAYOUT per M bucket, but a weight
    is stored in exactly one layout, so the config's layout is overridden with
    ``weight_layout``. Every native kernel (decode / prefill / splitK) reads
    either layout. DeepGEMM's large-M dequant + rocBLAS path only reads raw
    weights, so a marlin weight is dequantized in one pass by
    w8a16_marlin_dequant and fed to rocBLAS here. That path may run inside
    graph capture, so it bypasses DeepGEMM's dequant cache (a W16 cached
    during capture is never computed).
    """
    import deepgemm

    n, k = w.shape
    m = x.numel() // k
    config = deepgemm.get_w8a16_linear_config(m, n, k)
    kernel_type = int(config.get("KERNEL_TYPE", 0))
    if kernel_type == 2 and weight_layout == W8A16_LAYOUT_MARLIN:
        w16 = _w8a16_dequant(w, scale, x.dtype, weight_layout)
        return torch.nn.functional.linear(x, w16, bias)

    config = {**config, "WEIGHT_LAYOUT": weight_layout}
    out = deepgemm.w8a16_linear(x, w, scale, config=config)
    if bias is not None:
        out = out + bias
    return out


def _w8a16_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    weight_layout: int = W8A16_LAYOUT_RAW,
) -> torch.Tensor:
    """W8A16 GEMM with runtime backend selection."""
    use_deepgemm = (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_DEEPGEMM_W8A16_GEMM
    )
    if use_deepgemm:
        try:
            return _w8a16_gemm_deepgemm(x, w, scale, bias, weight_layout)
        except (ImportError, AttributeError):
            pass
    # Fallback to PyTorch native implementation
    if weight_layout == W8A16_LAYOUT_MARLIN:
        w = w8a16_unmarlin_weight(w)
    weight_scale = scale.unsqueeze(1)
    weights = w.view(torch.int8).to(x.dtype) * weight_scale.to(x.dtype)
    return torch.nn.functional.linear(x, weights, bias)


def _w8a16_gemm_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    weight_layout: int = W8A16_LAYOUT_RAW,
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], w.shape[0]), dtype=x.dtype, device=x.device
    )


try:
    direct_register_custom_op(
        op_name="w8a16_gemm",
        op_func=_w8a16_gemm,
        fake_impl=_w8a16_gemm_fake,
        tags=() if is_torch_equal_or_newer("2.7.0") else (
            torch.Tag.needs_fixed_stride_order,
        ),
    )
except RuntimeError:
    # Already registered by upstream or a previous load attempt
    pass


def w8a16_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    weight_layout: int = W8A16_LAYOUT_RAW,
) -> torch.Tensor:
    """Public entry: keep signature torch-compatible."""
    return torch.ops.vllm.w8a16_gemm(x, w, scale, bias, weight_layout)
