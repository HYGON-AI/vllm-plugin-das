# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import functools
import importlib
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


def _strided_b16_paged_fwd(parameters: dict[str, Any]):
    """Use the stride-aware D128 prefix kernel without copying the KV cache."""
    q, k, v = (parameters[name] for name in ("q", "k", "v"))
    output = parameters.get("out")
    native_output = (
        torch.empty_like(output, memory_format=torch.contiguous_format)
        if output is not None and not output.is_contiguous()
        else output
    )
    window = parameters.get("window_size") or (-1, -1)
    softmax_scale = parameters.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    cu = parameters["cu_seqlens_q"].contiguous()
    result = flash_attn_2_cuda.varlen_fwd(
        q.contiguous(), k, v, native_output, cu,
        # The cumulative K lengths are unused for paged attention.
        cu, parameters["seqused_k"].contiguous(), None,
        parameters["block_table"].contiguous(), parameters.get("alibi_slopes"),
        parameters["max_seqlen_q"], parameters["max_seqlen_k"],
        parameters.get("dropout_p", 0.0),
        softmax_scale,
        False, parameters.get("causal", False), window[0], window[1],
        parameters.get("softcap", 0.0), False,
        parameters.get("q_descale"), parameters.get("k_descale"),
        parameters.get("v_descale"), None, parameters.get("s_aux"),
    )
    native_output = result[0]
    if output is not None and native_output is not output:
        output.copy_(native_output)
    else:
        output = native_output
    if parameters.get("return_softmax_lse", False):
        return output, result[5], result[6]
    return output


def _softcap_b16_paged_fwd(parameters: dict[str, Any]):
    """Honor positive softcap without using the vendor's unsupported paged ABI."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    if parameters.get("q_v") is not None:
        raise NotImplementedError("HCU paged B16 softcap fallback does not support q_v")
    if parameters.get("cp_world_size", 1) != 1:
        raise NotImplementedError(
            "HCU paged B16 softcap fallback does not support context parallel attention"
        )
    if (
        parameters.get("dropout_p", 0.0) != 0.0
        or parameters.get("return_softmax_lse", False)
        or parameters.get("return_attn_probs", False)
    ):
        raise NotImplementedError(
            "HCU paged B16 softcap fallback does not support dropout or returning LSE/probabilities"
        )
    window = parameters.get("window_size") or (-1, -1)
    causal = parameters.get("causal", False)
    if not causal and window[1] >= 0:
        raise NotImplementedError(
            "HCU paged B16 softcap fallback does not support a noncausal right window"
        )
    q = parameters["q"].contiguous()
    output = parameters.get("out")
    contiguous_output = (
        output if output is not None and output.is_contiguous()
        else torch.empty_like(q)
    )
    scale = parameters.get("softmax_scale")
    unified_attention(
        q=q, k=parameters["k"], v=parameters["v"], out=contiguous_output,
        cu_seqlens_q=parameters["cu_seqlens_q"].contiguous(),
        max_seqlen_q=parameters["max_seqlen_q"],
        seqused_k=parameters["seqused_k"].contiguous(),
        max_seqlen_k=parameters["max_seqlen_k"],
        softmax_scale=q.shape[-1] ** -0.5 if scale is None else scale,
        causal=causal, window_size=window,
        block_table=parameters["block_table"].contiguous(),
        softcap=parameters["softcap"],
        q_descale=parameters.get("q_descale"),
        k_descale=parameters.get("k_descale"),
        v_descale=parameters.get("v_descale"),
        alibi_slopes=parameters.get("alibi_slopes"), sinks=parameters.get("s_aux"),
    )
    if output is None:
        return contiguous_output
    if contiguous_output is not output:
        output.copy_(contiguous_output)
    return output


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
        # vLLM's [block, K/V, token, head, channel] allocation interleaves
        # K/V pages. Its unbound caches are strided even in native NHD/HND
        # order. The gfx938 direct PA ABI rejects those views. The B16 D128
        # prefix kernel accepts their strides, so retain the original logical
        # bshd views instead of materializing the entire cache each decode.
        if (
            isinstance(key_cache, torch.Tensor)
            and key_cache.ndim == 4
            and key_cache.shape[1] == 64
            and key_cache.shape[-1] == 128
            and key_cache.dtype in (torch.bfloat16, torch.float16)
        ):
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            for parameter in signature.parameters.values():
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    parameters.update(parameters.pop(parameter.name, {}))
            q, v = parameters.get("q"), parameters.get("v")
            cu = parameters.get("cu_seqlens_q")
            max_q = parameters.get("max_seqlen_q", 0)
            if (
                isinstance(q, torch.Tensor) and q.ndim == 3
                and isinstance(v, torch.Tensor) and v.shape == key_cache.shape
                and q.shape[-1] == 128 and q.dtype == key_cache.dtype == v.dtype
                and isinstance(cu, torch.Tensor)
                and isinstance(parameters.get("block_table"), torch.Tensor)
                and isinstance(parameters.get("seqused_k"), torch.Tensor)
                and (
                    max_q == 1
                    or (
                        1 < max_q < 16
                        and max_q * (cu.shape[0] - 1) == q.shape[0]
                        and max_q * q.shape[1] <= 64 * key_cache.shape[2]
                    )
                )
            ):
                if parameters.get("softcap", 0.0) > 0.0:
                    # The native paged varlen ABI rejects positive softcap,
                    # while direct PA silently ignores it. Triton implements it.
                    return _softcap_b16_paged_fwd(parameters)
                if (
                    parameters.get("q_v") is None
                    and parameters.get("cp_world_size", 1) == 1
                    and not parameters.get("return_attn_probs", False)
                    and (
                        not key_cache.is_contiguous()
                        or not v.is_contiguous()
                        or not q.is_contiguous()
                        or (
                            isinstance(parameters.get("out"), torch.Tensor)
                            and not parameters["out"].is_contiguous()
                        )
                    )
                ):
                    return _strided_b16_paged_fwd(parameters)
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
