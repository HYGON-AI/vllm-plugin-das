# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import functools
import importlib
import inspect
from collections.abc import Callable
from math import prod
from typing import Any

import torch
import flash_attn_2_cuda
from flash_attn import (
    flash_attn_varlen_func as _flash_attn_varlen_func,
    hg_flash_attn_varlen_func as _hg_flash_attn_varlen_func,
    varlen_fwd_unified as _varlen_fwd_unified,
)
from torch import Tensor
from vllm_hcu.platforms import envs as hcu_envs
from vllm_hcu.v1.attention.kv_cache_layout import get_kv_cache_layout
from vllm_hcu.v1.attention._flash_attn_compat import (
    require_compatible_flash_attn as _require_compatible_flash_attn,
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
    parameter_positions = tuple(signature.parameters)
    layout_position = (
        parameter_positions.index("layout")
        if layout_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        else None
    )
    block_table_position = (
        parameter_positions.index("block_table")
        if "block_table" in signature.parameters
        else None
    )
    max_seqlen_q_position = (
        parameter_positions.index("max_seqlen_q")
        if "max_seqlen_q" in signature.parameters
        else None
    )

    descale_positions = {
        parameter: tuple(signature.parameters).index(parameter)
        for parameter in ("q_descale", "k_descale", "v_descale")
        if parameter in signature.parameters
        and signature.parameters[parameter].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        # Core expands layer-global scales to [batch, kv_heads]. The vendor
        # paged-attention ABI requires one FP32 value, including for 64-token
        # pages used by decode/MTP. Collapse only genuine scalar broadcasts:
        # taking the first element of per-sequence/head scales is incorrect.
        key_cache = args[1] if len(args) > 1 else kwargs.get("k")
        if isinstance(key_cache, torch.Tensor) and key_cache.ndim == 4:
            positional = list(args)
            for parameter in ("q_descale", "k_descale", "v_descale"):
                position = descale_positions.get(parameter)
                is_positional = position is not None and len(positional) > position
                descale = positional[position] if is_positional else kwargs.get(parameter)
                if (
                    isinstance(descale, torch.Tensor)
                    and descale.numel() > 1
                    and all(
                        size == 1 or stride == 0
                        for size, stride in zip(descale.shape, descale.stride())
                    )
                ):
                    scalar = descale.as_strided((1,), (1,))
                    if is_positional:
                        positional[position] = scalar
                    else:
                        kwargs[parameter] = scalar
            args = tuple(positional)
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
        # The HCU paged-attention ABI requires every PA tensor to be
        # contiguous. Keep the layout normalization above zero-copy, then
        # adapt the interleaved K/V pages and the small metadata tensors before
        # entering the vendor wrapper. Non-paged calls retain the original
        # call path and argument behavior.
        block_table = kwargs.get("block_table")
        if block_table is None and block_table_position is not None:
            if len(args) > block_table_position:
                block_table = args[block_table_position]
        max_seqlen_q = kwargs.get("max_seqlen_q")
        if max_seqlen_q is None and max_seqlen_q_position is not None:
            if len(args) > max_seqlen_q_position:
                max_seqlen_q = args[max_seqlen_q_position]
        if block_table is None or max_seqlen_q is None:
            return function(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        arguments = dict(bound.arguments)
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                arguments.update(arguments.pop(parameter.name, {}))
        return run_paged_attention(function, arguments)

    return wrapped


def _uses_direct_paged_attention(arguments: dict[str, Any]) -> bool:
    """Mirror the vendor's equal-head-dimension, 64-token PA dispatch."""
    if arguments.get("block_table") is None:
        return False
    q, k, v = (arguments.get(name) for name in ("q", "k", "v"))
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return False
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        return False
    layout = arguments.get("layout", "bshd")
    if layout not in ("bhsd", "bshd"):
        return False
    page_axis, head_axis = (2, 1) if layout == "bhsd" else (1, 2)
    if k.shape[page_axis] != 64 or q.shape[-1] != v.shape[-1]:
        return False
    max_q = arguments.get("max_seqlen_q")
    if max_q == 1:
        return True
    cu_seqlens_q = arguments.get("cu_seqlens_q")
    return (
        max_q is not None
        and 1 < max_q < 16
        and isinstance(cu_seqlens_q, torch.Tensor)
        and max_q * (cu_seqlens_q.shape[0] - 1) == q.shape[0]
        and max_q * q.shape[1] <= 64 * k.shape[head_axis]
    )


def _page_view(cache: torch.Tensor) -> tuple[torch.Tensor, int | None]:
    """Expose dense virtual pages for an interleaved K/V cache view.

    vLLM's fused cache stores K and V pages next to each other. The logical
    K/V views therefore have a page stride of two pages even though each page
    is dense internally. The vendor PA ABI only accepts a contiguous page
    tensor, so expose the intervening pages as a zero-copy view and remap the
    block table to the even/odd virtual page IDs. The cache allocation and
    writer views remain unchanged.
    """
    if cache.is_contiguous():
        return cache, 1
    if cache.ndim != 4 or cache.shape[0] == 0 or not cache[0].is_contiguous():
        raise ValueError("HCU PA requires dense contents within each KV page")
    page_elements = prod(cache.shape[1:])
    if cache.stride(0) < page_elements:
        raise ValueError("HCU PA KV pages overlap")
    if cache.stride(0) % page_elements:
        return cache, None
    ratio = cache.stride(0) // page_elements
    # Do not expose the trailing unused half-page after the final real page.
    shape = ((cache.shape[0] - 1) * ratio + 1, *cache.shape[1:])
    view = torch.as_strided(cache, shape, (page_elements, *cache.stride()[1:]))
    return view, ratio


def _run_stride_aware_paged_attention(arguments: dict[str, Any]) -> Any:
    """Use Core's stride-aware FP8 kernel without copying the allocated cache."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    if (
        arguments.get("dropout_p", 0.0) != 0.0
        or arguments.get("return_softmax_lse", False)
        or arguments.get("return_attn_probs", False)
        or arguments.get("q_v") is not None
        or arguments.get("cp_world_size", 1) != 1
    ):
        raise NotImplementedError(
            "HCU stride-aware PA fallback requires local inference without LSE/probabilities"
        )
    q, k, v = (arguments[name] for name in ("q", "k", "v"))
    if arguments.get("layout", "bshd") == "bhsd":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    window = arguments.get("window_size") or (-1, -1)
    causal = arguments.get("causal", False)
    if not causal and (window[0] >= 0 or window[1] >= 0):
        raise NotImplementedError("HCU stride-aware PA requires global noncausal attention")
    slopes = arguments.get("alibi_slopes")
    if slopes is not None and slopes.ndim != 1:
        raise NotImplementedError("HCU stride-aware PA requires per-head ALiBi slopes")
    quantized = k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    scales = {}
    for name in ("q_descale", "k_descale", "v_descale"):
        scale = arguments.get(name)
        if scale is not None and scale.numel() != 1:
            raise NotImplementedError("HCU stride-aware PA requires scalar descales")
        if scale is None and quantized:
            scale = torch.ones(1, device=q.device, dtype=torch.float32)
        scales[name] = scale
    # BF16 Q also avoids the native FP8 probability-rounding error and works
    # for E5M2 queries when Core's platform default FP8 format is E4M3.
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        q = (q.float() * scales["q_descale"]).to(torch.bfloat16)
        scales["q_descale"] = None
    output = arguments.get("out")
    dtype = torch.bfloat16 if q.dtype in (
        torch.float8_e4m3fn, torch.float8_e5m2,
    ) else q.dtype
    native_output = (
        output if output is not None and output.is_contiguous()
        else torch.empty(q.shape, device=q.device, dtype=dtype)
    )
    scale = arguments.get("softmax_scale")
    unified_attention(
        q=q.contiguous(), k=k, v=v, out=native_output,
        cu_seqlens_q=arguments["cu_seqlens_q"].contiguous(),
        max_seqlen_q=arguments["max_seqlen_q"],
        seqused_k=arguments["seqused_k"].contiguous(),
        max_seqlen_k=arguments["max_seqlen_k"],
        softmax_scale=q.shape[-1] ** -0.5 if scale is None else scale,
        causal=causal, window_size=window,
        block_table=arguments["block_table"].contiguous(),
        softcap=arguments.get("softcap", 0.0), alibi_slopes=slopes,
        sinks=arguments.get("s_aux"),
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR if quantized else KVQuantMode.NONE,
        **scales,
    )
    if output is not None and output is not native_output:
        output.copy_(native_output)
        return output
    return native_output


def run_paged_attention(
    vendor: Callable[..., Any], kwargs: dict[str, Any]
) -> Any:
    """Adapt decode and MTP calls to the contiguous vendor PA ABI."""
    key_cache = kwargs.get("k")
    if kwargs.get("block_table") is not None and isinstance(key_cache, torch.Tensor):
        kv_heads = key_cache.shape[1 if kwargs.get("layout") == "bhsd" else 2]
        # PR #551 also excludes this E4M3 geometry from native PA: its
        # contiguous control fails non-unit-scale accuracy qualification.
        # Core's FP8 kernel keeps the original strided pages and applies scales.
        if key_cache.dtype == torch.float8_e4m3fn and (
            key_cache.shape[-1] == 256 and kv_heads == 4
        ):
            return _run_stride_aware_paged_attention(kwargs)
    # Match the vendor's 64-token direct-PA admission, including uniform MTP.
    if not _uses_direct_paged_attention(kwargs):
        return vendor(**kwargs)

    arguments = dict(kwargs)
    key, key_ratio = _page_view(arguments["k"])
    value, value_ratio = _page_view(arguments["v"])
    if key_ratio != value_ratio:
        raise ValueError("HCU PA requires K and V to have the same page stride")
    if key_ratio is None or max(
        key.numel() * key.element_size(), value.numel() * value.element_size(),
    ) > torch.iinfo(torch.int32).max:
        # Virtual pages double the addressable span for fused K/V storage.
        # Avoid the vendor's signed-32-bit indexing limit and never copy the
        # full cache. Core uses stride-aware 64-bit cache addressing instead.
        return _run_stride_aware_paged_attention(kwargs)
    arguments.update(k=key, v=value)

    block_table = arguments["block_table"]
    if key_ratio is not None and key_ratio != 1:
        block_table = block_table * key_ratio
    arguments["block_table"] = block_table.contiguous()

    for name in (
        "q",
        "seqused_k",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "alibi_slopes",
        "q_descale",
        "k_descale",
        "v_descale",
    ):
        tensor = arguments.get(name)
        if tensor is not None:
            arguments[name] = tensor.contiguous()

    output = arguments.get("out")
    if output is not None and not output.is_contiguous():
        arguments["out"] = torch.empty_like(
            output, memory_format=torch.contiguous_format
        )


    result = vendor(**arguments)
    if output is not None and arguments.get("out") is not output:
        output.copy_(arguments["out"])
        if isinstance(result, tuple):
            return (output, *result[1:])
        return output
    return result


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
