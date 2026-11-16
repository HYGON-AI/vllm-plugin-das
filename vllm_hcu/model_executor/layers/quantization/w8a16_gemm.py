# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main (0.18) modelopt.py w8a16 stack: deepgemm W8A16
# GEMM with tuned-config layout awareness (marlin pre-shuffle support), a
# one-pass marlin dequant Triton kernel for the large-M path, and a PyTorch
# dequant fallback, registered as torch.ops.vllm.w8a16_gemm (the registration
# point the 0.18 overlay owned).

from __future__ import annotations

import functools
import os
from collections import OrderedDict

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


# Distinct (N, K) pairs already reported as pre-shuffled this process.
_LOGGED_PRESHUFFLE_SHAPES: set = set()


def preshuffle_linear_weight(weight: torch.Tensor, scheme: str) -> tuple:
    """Pre-shuffle an int8 [N, K] weight to marlin layout if tuned configs
    want it; returns (weight, W8A16_LAYOUT_*). Shared by the ModelOpt and
    compressed-tensors W8A16 linear methods so the decision logic lives in
    exactly one place."""
    import vllm_hcu.platforms.envs as _henvs

    if not _henvs.VLLM_HCU_USE_CUSTOM_OPS:
        return weight, W8A16_LAYOUT_RAW
    if not w8a16_should_preshuffle(*weight.shape):
        return weight, W8A16_LAYOUT_RAW
    import deepgemm

    shape = tuple(weight.shape)
    if shape not in _LOGGED_PRESHUFFLE_SHAPES:
        _LOGGED_PRESHUFFLE_SHAPES.add(shape)
        # warning, not info: vllm_hcu.* loggers run at WARNING inside
        # EngineCore, an info line here would be silently dropped.
        logger.warning(
            "W8A16(%s): pre-shuffling linear weights into DeepGEMM marlin "
            "layout (deepgemm %s), (N, K)=%s",
            scheme, _deepgemm_version(), shape)
    return deepgemm.w8a16_marlin_weight(weight), W8A16_LAYOUT_MARLIN


def w8a16_unmarlin_weight(w: torch.Tensor) -> torch.Tensor:
    """Inverse of deepgemm.w8a16_marlin_weight: marlin -> row-major [N, K]."""
    n, k = w.shape
    assert n % 16 == 0 and k % 128 == 0, (
        f"unmarlin tile geometry requires N%16==0 and K%128==0, got {(n, k)}")
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


# HCU-measured DeepGEMM configs for marlin-layout weights where the bundled
# table (tuned for the raw layout) is slow: (N, K) -> [(max_m, config)].
# K=4096 GEMV (GDN out_proj / attention o_proj): 21.0 -> 16.2 us at M=1,
# 24.7 -> 18.2 us at M=8.
_SPLITK_K4096 = dict(NUM_K_WARPS=4, N_PER_WARP=32, K_PER_WARP=128,
                     NUM_N_WARPS=2, BLOCK_M=16, KERNEL_TYPE=0)
_MARLIN_CONFIG_OVERRIDES = {
    (2048, 4096): ((1, {**_SPLITK_K4096, "SPLITK": 8}),
                   (8, {**_SPLITK_K4096, "SPLITK": 4})),
}


def _config_override(m: int, n: int, k: int, weight_layout: int):
    if weight_layout != W8A16_LAYOUT_MARLIN:
        return None
    for max_m, cfg in _MARLIN_CONFIG_OVERRIDES.get((n, k), ()):
        if m <= max_m:
            return cfg
    return None


# (weight ptr, weight version, scale ptr, scale version, dtype, layout)
# -> W16 weight.  The key carries the storage versions because a weight can be
# rewritten in place (reload, adapter swap) -- shape/dtype alone would then
# return the previous model's tensor.  dtype and weight_layout are both inputs
# to _w8a16_dequant, so they belong in the key too.  Entries are dropped
# wholesale by clear_dequant_cache(), which HcuGPUWorker.reload_weights calls;
# that, not the bound below, is what keeps the cache from growing reload over
# reload.
#
# The bound is only a runaway guard and is deliberately far above any real
# working set (35B reaches ~240 entries: one per dense W8A16 linear, the MoE
# experts take the AITER path instead).  It must stay that way: a forward pass
# touches weights cyclically in layer order, which is the pattern LRU handles
# worst -- if the bound ever fell below the working set the hit rate would go
# to zero and every call would re-dequantize.  Eviction therefore warns.
_MAX_CACHED_WEIGHTS = 4096
_WARNED_EVICTION = [False]
_CacheKey = tuple  # (wptr, wver, sptr, sver, dtype, layout)
_DEQUANT_CACHE: "OrderedDict[_CacheKey, torch.Tensor]" = OrderedDict()
# Inputs per key, so the storage stays alive (and data_ptr cannot be reused
# by an unrelated tensor) for as long as the entry exists.
_DEQUANT_CACHE_SRC: dict[_CacheKey,
                         tuple[torch.Tensor, torch.Tensor]] = {}


def clear_dequant_cache() -> None:
    """Drop every cached dequantized weight.

    Call after weights change identity (sleep/wake, adapter swap, weight
    reload): the int8/W16 pair can be ~2.6 GB on a 35B model, and a stale
    entry would otherwise outlive the weights it was derived from.
    """
    _DEQUANT_CACHE.clear()
    _DEQUANT_CACHE_SRC.clear()


def _cached_dequant(w, scale, dtype, weight_layout):
    """Large-M path: reuse a dequantized copy instead of re-dequantizing every
    call (~2x faster prefill GEMMs) at the cost of a W16 copy per weight.

    On by default; VLLM_HCU_W8A16_CACHE_DEQUANT=0 disables it.  The copy is
    only created outside CUDA-graph capture (the eager profile run reaches
    this path with max_num_batched_tokens, so every layer is cached before
    capture and the memory is accounted for before the KV cache is sized).

    The key includes ``w._version`` and the scale's pointer/version: comparing
    shape and dtype alone would hand back the previous model's weights when
    the same storage is rewritten in place (reload, adapter swap), producing
    stale logits without any error.
    """
    if not henvs.optional_custom_op_enabled(henvs.VLLM_HCU_W8A16_CACHE_DEQUANT):
        return None
    key = (w.data_ptr(), w._version, scale.data_ptr(), scale._version,
           dtype, weight_layout)
    hit = _DEQUANT_CACHE.get(key)
    if hit is not None:
        _DEQUANT_CACHE.move_to_end(key)
        return hit
    if torch.cuda.is_current_stream_capturing():
        return None
    w16 = _w8a16_dequant(w, scale, dtype, weight_layout)
    _DEQUANT_CACHE[key] = w16
    _DEQUANT_CACHE_SRC[key] = (w, scale)
    while len(_DEQUANT_CACHE) > _MAX_CACHED_WEIGHTS:
        if not _WARNED_EVICTION[0]:
            _WARNED_EVICTION[0] = True
            logger.warning(
                "W8A16 dequant cache exceeded %d entries and is evicting. If "
                "the working set is larger than the bound every call will "
                "re-dequantize; set VLLM_HCU_W8A16_CACHE_DEQUANT=0 or raise "
                "the bound.", _MAX_CACHED_WEIGHTS)
        stale, _ = _DEQUANT_CACHE.popitem(last=False)
        _DEQUANT_CACHE_SRC.pop(stale, None)
    return w16


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
        w16 = _cached_dequant(w, scale, x.dtype, weight_layout)
        if w16 is None:
            w16 = _w8a16_dequant(w, scale, x.dtype, weight_layout)
        return torch.nn.functional.linear(x, w16, bias)

    override = _config_override(m, n, k, weight_layout)
    config = {**(override or config), "WEIGHT_LAYOUT": weight_layout}
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
