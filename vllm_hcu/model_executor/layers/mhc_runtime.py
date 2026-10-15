# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""AITER mHC adapters, including the previous-sublayer Pre used by V4.1.

Keep the unfused decoder RMSNorm contract from SGLang a0840f4. TileLang
Pre uses the compact projection layout required by SGLang's mhc.py.
"""
from __future__ import annotations

import functools
import importlib
import logging
from dataclasses import dataclass
from typing import Callable

import torch

from vllm_hcu.platforms import envs as henvs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MhcRuntime:
    pre: Callable | None = None
    post: Callable | None = None
    fused: Callable | None = None
    project: Callable | None = None
    big_fuse: Callable | None = None
    tilelang_pre: Callable | None = None
    tilelang_post: Callable | None = None

    @property
    def delayed_available(self) -> bool:
        return self.project is not None and (
            self.big_fuse is not None or self.tilelang_pre is not None
        )


def _operator(module, name):
    value = getattr(module, name, None)
    return value if callable(value) else None


@functools.lru_cache(maxsize=2)
def get_runtime(tilelang: bool = False) -> MhcRuntime:
    try:
        native = importlib.import_module("aiter.ops.mhc")
    except (ImportError, AttributeError):
        native = None
    tile = None
    if tilelang:
        try:
            tile = importlib.import_module("aiter.ops.tilelang")
        except (ImportError, AttributeError, RuntimeError) as exc:
            # A discovered wheel need not have a usable TVM/TileLang runtime.
            logger.warning("AITER TileLang mHC unavailable; using HIP fallback: %s", exc)
    return MhcRuntime(
        pre=_operator(native, "mhc_pre"),
        post=_operator(native, "mhc_post"),
        fused=_operator(native, "mhc_fused_post_pre"),
        project=_operator(native, "mhc_pre_gemm_sqrsum"),
        big_fuse=_operator(native, "mhc_pre_big_fuse"),
        tilelang_pre=_operator(tile, "pre_big_fuse_tilelang"),
        tilelang_post=_operator(tile, "mhc_post_fwd"),
    )


def runtime() -> MhcRuntime:
    return get_runtime(henvs.VLLM_HCU_USE_AITER_TILELANG_MHC)


def pre_enabled() -> bool:
    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_USE_AITER_MHC and henvs.VLLM_HCU_USE_AITER_MHC_PRE
    )


def pre_supported(residual, fn, hc_scale, hc_base, pre_mix=None, x=None) -> bool:
    if residual.ndim != 3 or residual.shape[1] != 4:
        return False
    tokens, hc, hidden = residual.shape
    width = hc * hidden if x is None else hidden
    tensors = [residual, fn, hc_scale, hc_base]
    if pre_mix is not None:
        if pre_mix.shape != (tokens, hc) or pre_mix.dtype != torch.float32:
            return False
        tensors.append(pre_mix)
    if x is not None:
        if x.shape != (tokens, hidden) or x.dtype != torch.bfloat16:
            return False
        tensors.append(x)
    return (
        hidden >= 1024
        and hidden % 1024 == 0
        and residual.dtype == torch.bfloat16
        and fn.shape == (hc * (2 + hc), width)
        and hc_scale.shape == (3,)
        and hc_base.shape == (hc * (2 + hc),)
        and fn.dtype == hc_scale.dtype == hc_base.dtype == torch.float32
        and all(t.is_cuda and t.device == residual.device
                and t.is_contiguous() for t in tensors)
    )


def _empty_pre(residual):
    tokens, hc, hidden = residual.shape
    options = {"device": residual.device, "dtype": torch.float32}
    return (
        torch.empty(tokens, hc, 1, **options),
        torch.empty(tokens, hc, hc, **options),
        torch.empty(tokens, hidden, device=residual.device, dtype=residual.dtype),
        torch.empty(tokens, hc, **options),
    )


def _staged_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                n_splits=1, x=None):
    ops = runtime()
    if not ops.delayed_available:
        raise RuntimeError("AITER staged mHC Pre is unavailable")
    tokens, hc, hidden = residual.shape
    if tokens == 0:
        post, comb, layer_input, _ = _empty_pre(residual)
        empty_gemm = torch.empty(1, 0, hc * (2 + hc), device=residual.device)
        empty_sqr = torch.empty(1, 0, device=residual.device)
        return post, comb, layer_input, empty_gemm, empty_sqr
    projected = residual
    if x is not None:
        # The first layer projects a single embedding, not the 4-stream buffer.
        # Repeating it preserves its mean-square; only stream 0 carries weights.
        projected = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
        expanded_fn = fn.new_zeros(fn.shape[0], hc * hidden)
        expanded_fn[:, :hidden].copy_(fn)
        fn = expanded_fn
    tile_k = 64
    if n_splits < 1 or (hc * hidden) % (n_splits * tile_k):
        raise ValueError("mHC split-K must divide the projection width")
    if hc * hidden // n_splits < 2 * tile_k:
        raise ValueError("mHC split-K is too large for the projection width")
    padded = torch.empty(n_splits, tokens, 32, dtype=torch.float32,
                         device=residual.device)
    gemm = padded[:, :, :hc * (2 + hc)]
    sqrsum = torch.empty(n_splits, tokens, dtype=torch.float32,
                         device=residual.device)
    post, comb, layer_input, _ = _empty_pre(residual)
    ops.project(gemm, sqrsum, projected, fn, tile_k, False)
    if ops.tilelang_pre is not None:
        # SGLang compacts 32-column split-K padding before AITER TileLang Pre.
        ops.tilelang_pre(
            gemm.contiguous(), sqrsum, hc_scale, hc_base, residual,
            post.squeeze(-1), comb.flatten(1), layer_input, hidden, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            n_splits=n_splits, mhc_mult=hc,
        )
    else:
        ops.big_fuse(
            post, comb, layer_input, gemm, sqrsum, hc_scale, hc_base, residual,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat,
        )
    return post, comb, layer_input, gemm, sqrsum


def pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1):
    ops = runtime()
    if residual.shape[0] == 0:
        return _empty_pre(residual)[:3]
    if ops.tilelang_pre is not None and ops.delayed_available:
        return _staged_pre(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
        )[:3]
    if ops.pre is None:
        raise RuntimeError("AITER mHC Pre is unavailable")
    with torch.device(residual.device):
        return ops.pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)


def delayed_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.shape[0] == 0:
        return _empty_pre(residual)
    # Conservative HCU layout; do not depend on gfx95-only AITER tuning helpers.
    post, comb, _, gemm, sqrsum = _staged_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, x=x,
    )
    hc, hidden = residual.shape[1:]
    next_pre = torch.ops.vllm.mhc_pre_mix_triton(
        gemm, sqrsum, hc_scale, hc_base, hc, hc * hidden, rms_eps, hc_pre_eps,
    )
    if pre_mix is None:
        # Return fresh storage: custom-op outputs may not alias input buffers.
        layer_input = residual[:, 0].clone()
    else:
        layer_input = torch.ops.vllm.hc_collapse_triton(residual, pre_mix)
    return post, comb, layer_input, next_pre


def delayed_pre_fake(
    residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
    hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None, x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _empty_pre(residual)


_REGISTERED = False


def register_delayed_op() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="hcu_mhc_pre_delayed", op_func=delayed_pre,
        mutates_args=[], fake_impl=delayed_pre_fake,
    )
    _REGISTERED = True


def post(x, residual, post_layer_mix, comb_res_mix):
    ops = runtime()
    if residual.shape[0] == 0:
        return torch.empty_like(residual)
    if ops.tilelang_post is not None and residual.dtype == torch.bfloat16:
        return ops.tilelang_post(x, residual, post_layer_mix.reshape(-1, 4),
                                 comb_res_mix)
    if ops.post is None:
        raise RuntimeError("AITER mHC Post is unavailable")
    out = torch.empty_like(residual)
    ops.post(out, x, residual, post_layer_mix.reshape(-1, 4, 1), comb_res_mix)
    return out


def fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                   hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                   hc_post_mult_value, sinkhorn_repeat):
    ops = runtime()
    if residual.shape[0] == 0:
        return torch.empty_like(residual), *_empty_pre(residual)[:3]
    if henvs.VLLM_HCU_USE_AITER_MHC_FUSED and ops.fused is not None:
        # AITER order differs from vLLM. Norm is deliberately applied by caller.
        with torch.device(residual.device):
            post_mix, comb_mix, layer_input, updated = ops.fused(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat,
            )
        return updated, post_mix, comb_mix, layer_input
    updated = post(x, residual, post_layer_mix, comb_res_mix)
    return updated, *pre(updated, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
