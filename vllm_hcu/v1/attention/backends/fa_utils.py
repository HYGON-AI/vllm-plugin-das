# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import functools
import importlib.metadata as importlib_metadata
import inspect
from collections.abc import Callable
from typing import Any

import torch
from flash_attn import (
    flash_attn_varlen_func as _flash_attn_varlen_func,
    hg_flash_attn_varlen_func as _hg_flash_attn_varlen_func,
    varlen_fwd_unified as _varlen_fwd_unified,
    vllm_flash_attn_varlen_func,
)
from torch import Tensor
from vllm_hcu.v1.attention.kv_cache_layout import get_kv_cache_layout

import vllm_hcu.hcu_ops as hcu_ops


_FLASH_ATTN_BUILDS_WITH_NATIVE_LONG_PREFILL_GATHER = frozenset(
    {"2.8.4+dtk2604.torch2110.2609241509.g624d7b"}
)


def _require_native_long_prefill_gather() -> None:
    try:
        actual = importlib_metadata.version("flash_attn")
    except importlib_metadata.PackageNotFoundError:
        actual = None
    if actual not in _FLASH_ATTN_BUILDS_WITH_NATIVE_LONG_PREFILL_GATHER:
        raise RuntimeError(
            "HCU requires a flash_attn build with the native long-prefill "
            "gather fix; supported builds: "
            f"{sorted(_FLASH_ATTN_BUILDS_WITH_NATIVE_LONG_PREFILL_GATHER)}; "
            f"found {actual!r}. Rebuild the image with a supported vendor wheel."
        )


_require_native_long_prefill_gather()


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
            # vLLM exposes logical [block, token, head, channel] views even
            # when the backing storage is HND. The vendor bhsd interface
            # also requires the head/token axes in that order in the shape.
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
    torch.ops.hcu_ops.reshape_and_cache_flash(
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
