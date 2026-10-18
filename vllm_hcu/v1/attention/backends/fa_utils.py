# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import functools
import importlib
import importlib.metadata as importlib_metadata
import inspect
from collections.abc import Callable
from typing import Any

import torch
import flash_attn_2_cuda
from flash_attn import (
    flash_attn_varlen_func as _flash_attn_varlen_func,
    hg_flash_attn_varlen_func as _hg_flash_attn_varlen_func,
    varlen_fwd_unified as _varlen_fwd_unified,
)
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from torch import Tensor
from vllm_hcu.platforms import envs as hcu_envs
from vllm_hcu.v1.attention.kv_cache_layout import get_kv_cache_layout



# This range describes the Python/native API line exercised by the plugin. It
# does not attest that a particular vendor artifact contains the native
# long-prefill gather workspace fix; deployment images must validate that
# binary capability separately.
_FLASH_ATTN_COMPATIBLE_VERSIONS = SpecifierSet(
    ">=2.8.4,<2.9", prereleases=False
)


def _require_compatible_flash_attn() -> None:
    try:
        actual = importlib_metadata.version("flash_attn")
    except importlib_metadata.PackageNotFoundError:
        actual = None

    try:
        parsed = None if actual is None else Version(actual)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Invalid flash_attn distribution version: {actual!r}"
        ) from exc

    if parsed is None or parsed not in _FLASH_ATTN_COMPATIBLE_VERSIONS:
        raise RuntimeError(
            "HCU requires a compatible flash_attn release "
            f"({_FLASH_ATTN_COMPATIBLE_VERSIONS}); found {actual!r}. "
            "The deployed vendor wheel must separately be validated for "
            "native long-prefill gather."
        )


_require_compatible_flash_attn()


def _flash_attn_layout() -> str:
    cache_layout = get_kv_cache_layout()
    if cache_layout == "HND":
        return "bhsd"
    if cache_layout == "NHD":
        return "bshd"
    raise ValueError(f"Unknown cache layout format {cache_layout}.")


def _with_kv_cache_layout(function: Callable[..., Any], name: str):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"HCU requires flash_attn.{name} to expose a layout parameter"
        ) from exc
    layout_parameter = signature.parameters.get("layout")
    if layout_parameter is None or layout_parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise RuntimeError(
            f"HCU requires flash_attn.{name} to expose a keyword-compatible "
            "layout parameter"
        )
    layout_position = (
        tuple(signature.parameters).index("layout")
        if layout_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        else None
    )

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        layout = _flash_attn_layout()
        if layout == "bhsd":
            key_cache = args[1] if len(args) > 1 else kwargs.get("k")
            # The gfx936 vendor varlen_fwd ABI does not receive pa_layout.
            # For 128-token pages it consequently reads dim 1 as the page
            # size. Keep the zero-copy logical bshd view so dim 1 remains the
            # page axis; 64-token pages use the native bhsd paged/gather path.
            use_bshd_compat = (
                isinstance(key_cache, torch.Tensor)
                and key_cache.ndim == 4
                and key_cache.shape[1] == 128
            )
            if use_bshd_compat:
                layout = "bshd"
                # vLLM expands each layer-global KV scale to [batch, heads],
                # while this vendor path accepts exactly one scalar. Recover
                # a zero-copy scalar view of the shared backing value.
                for descale_name in ("q_descale", "k_descale", "v_descale"):
                    descale = kwargs.get(descale_name)
                    if isinstance(descale, torch.Tensor) and descale.numel() > 1:
                        kwargs[descale_name] = descale.as_strided(
                            (1,), (1,), descale.storage_offset()
                        )
                # The vendor compatibility path casts BF16 queries directly
                # to the FP8 cache dtype before dispatch. That cast has an
                # implicit unit scale, but the kernel requires all three
                # descale tensors whenever K/V descales are present.
                if (
                    key_cache.dtype == torch.float8_e5m2
                    and kwargs.get("q_descale") is None
                    and isinstance(kwargs.get("k_descale"), torch.Tensor)
                ):
                    kwargs["q_descale"] = torch.ones_like(kwargs["k_descale"])
            # vLLM exposes logical [block, token, head, channel] views even
            # when the backing storage is HND. The vendor bhsd interface
            # also requires the head/token axes in that order in the shape.
            else:
                positional = list(args)
                for position, parameter in ((1, "k"), (2, "v")):
                    cache = (
                        positional[position]
                        if len(positional) > position
                        else kwargs.get(parameter)
                    )
                    if isinstance(cache, torch.Tensor) and cache.ndim == 4:
                        native_view = cache.transpose(1, 2)
                        if len(positional) > position:
                            positional[position] = native_view
                        else:
                            kwargs[parameter] = native_view
                args = tuple(positional)
        if layout_position is not None and len(args) > layout_position:
            positional = list(args)
            positional[layout_position] = layout
            args = tuple(positional)
            kwargs.pop("layout", None)
        else:
            kwargs["layout"] = layout
        return function(*args, **kwargs)

    return wrapped


# The target vendor FlashAttention owns paged-to-contiguous workspace sizing
# and uses its native gather. Keep block_table and seqused_k on that path to
# avoid a slower plugin-side gather for long chunked prefill.
flash_attn_varlen_func = _with_kv_cache_layout(
    _flash_attn_varlen_func,
    "flash_attn_varlen_func",
)
hg_flash_attn_varlen_func = _with_kv_cache_layout(
    _hg_flash_attn_varlen_func,
    "hg_flash_attn_varlen_func",
)
varlen_fwd_unified = _with_kv_cache_layout(
    _varlen_fwd_unified,
    "varlen_fwd_unified",
)


def can_use_paged_fp8_d192_fwd(
    key_cache: Tensor,
    value_cache: Tensor,
    out: Tensor,
    alibi_slopes: Tensor | None,
    softcap: float,
    s_aux: Tensor | None,
) -> bool:
    """Whether the paged FP8 head_dim-192 HG prefix kernel can serve a call."""
    return (
        key_cache.dtype == torch.float8_e4m3fn
        and value_cache.dtype == torch.float8_e4m3fn
        and key_cache.ndim == 4
        and key_cache.shape[-1] == 192
        and value_cache.shape[-1] == 192
        and key_cache.shape[1] in (64, 128)
        and out.dtype == torch.bfloat16
        and alibi_slopes is None
        and not softcap
        and (s_aux is None or s_aux.dtype == out.dtype)
        and get_kv_cache_layout() == "NHD"
    )


def paged_fp8_d192_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    block_table: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: list[int] | None,
    s_aux: Tensor | None = None,
) -> Tensor:
    """FP8 paged attention for head_dim 192 through the HG prefix kernel.

    varlen_fwd_unified sends this shape to two unsuitable kernels: decode
    (q_len <= 4) goes to paged_attention, which quantizes the per-partition
    *normalized* probabilities to e4m3 (most fall into the subnormal range,
    15-30% output error), and prefill goes to a contiguous FP8 path that only
    supports head_dim 128. The C++ varlen_fwd on the paged cache serves both
    with the HG prefix kernel (~2% error vs an exact reference on the same
    FP8 bytes), and only reads device-side lengths, so it is graph-safe.
    """
    window_left, window_right = (
        (window_size[0], window_size[1]) if window_size is not None else (-1, -1)
    )
    flash_attn_2_cuda.varlen_fwd(
        q.to(k.dtype),
        k,
        v,
        out,
        cu_seqlens_q,
        # cu_seqlens_k is unused with seqused_k but must be a tensor.
        cu_seqlens_q,
        seqused_k,
        None,  # leftpad_k
        block_table,
        None,  # alibi_slopes
        max_seqlen_q,
        max_seqlen_k,
        0.0,  # dropout
        softmax_scale,
        False,  # zero_tensors
        causal,
        window_left,
        window_right,
        0.0,  # softcap
        False,  # return_softmax
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        None,  # generator
        s_aux,
    )
    return out


# Hcu doesn't use scheduler metadata (FA3 feature), provide stub
def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
    return None


def reshape_and_cache_flash(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None:
    """Write FlashAttention KV pages using the active physical layout.

    The native HCU writer follows vLLM's stride-aware cache contract for every
    supported cache dtype and both NHD and HND storage. Keep cache writes
    independent of the vendor AITER package; AITER is an MoE backend here.
    """
    operator_kv_cache_dtype = (
        "auto"
        if kv_cache_dtype in {"bfloat16", "float16"}
        else kv_cache_dtype
    )
    if hcu_envs.custom_ops_enabled():
        try:
            importlib.import_module("vllm_hcu.hcu_ops")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HCU FlashAttention cache writer requires the vllm_hcu.hcu_ops "
                "native extension; rebuild or reinstall vllm_hcu so the "
                "extension matches this source checkout"
            ) from exc
        cache_writer = getattr(
            getattr(torch.ops, "hcu_ops", None),
            "reshape_and_cache_flash",
            None,
        )
        if not callable(cache_writer):
            raise RuntimeError(
                "HCU FlashAttention cache writer requires "
                "torch.ops.hcu_ops.reshape_and_cache_flash; rebuild or "
                "reinstall vllm_hcu so the native extension matches this "
                "source checkout"
            )
    else:
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        cache_writer = triton_reshape_and_cache_flash

    cache_writer(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        operator_kv_cache_dtype,
        k_scale,
        v_scale,
    )


def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    return 2


def flash_attn_supports_fp8() -> bool:
    return True


def flash_attn_supports_sinks() -> bool:
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True


def flash_attn_supports_quant_query_input() -> bool:
    return True


def is_fa_version_supported(fa_version: int) -> bool:
    return False
