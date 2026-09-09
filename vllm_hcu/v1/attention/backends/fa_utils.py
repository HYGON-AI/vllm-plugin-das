# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import functools
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
from vllm.v1.attention.backends.utils import get_kv_cache_layout

import vllm_hcu.hcu_ops as hcu_ops

_DSPARK_MAX_CAPTURE_TOKENS = 512
_DSPARK_MAX_CONTEXT_LENGTH = 4096
_DSPARK_MAX_STATIC_KV_SCRATCH_BYTES = 584 * 1024**2


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
        if layout_position is not None and len(args) > layout_position:
            positional = list(args)
            positional[layout_position] = layout
            args = tuple(positional)
            kwargs.pop("layout", None)
        else:
            kwargs["layout"] = layout
        return function(*args, **kwargs)

    return wrapped


def _matches_dspark_attention_shape(
    kwargs: dict[str, Any],
    *,
    query_len: int,
    causal: bool,
) -> bool:
    """Match a DSpark-7 attention shape with capture-safe static metadata."""
    q = kwargs.get("q")
    k = kwargs.get("k")
    v = kwargs.get("v")
    out = kwargs.get("out")
    cu_seqlens_q = kwargs.get("cu_seqlens_q")
    seqused_k = kwargs.get("seqused_k")
    block_table = kwargs.get("block_table")
    window_size = kwargs.get("window_size", (-1, -1))

    if not all(isinstance(tensor, Tensor) for tensor in (q, k, v, out)):
        return False
    if not all(
        isinstance(tensor, Tensor)
        for tensor in (cu_seqlens_q, seqused_k, block_table)
    ):
        return False
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        return False
    if out.shape != q.shape:
        return False
    tensors = (q, k, v, out, cu_seqlens_q, seqused_k, block_table)
    if any(tensor.device != q.device for tensor in tensors):
        return False
    contiguous_tensors = (q, out, cu_seqlens_q, seqused_k, block_table)
    if any(not tensor.is_contiguous() for tensor in contiguous_tensors):
        return False
    if kwargs.get("layout") != "bshd":
        return False
    if kwargs.get("max_seqlen_q") != query_len:
        return False
    if cu_seqlens_q.ndim != 1:
        return False
    batch_size = cu_seqlens_q.shape[0] - 1
    if batch_size <= 0 or q.shape[0] != batch_size * query_len:
        return False
    if block_table.ndim != 2 or block_table.shape[0] != batch_size:
        return False
    if seqused_k.ndim != 1 or seqused_k.shape[0] != batch_size:
        return False
    if any(
        tensor.dtype != torch.int32
        for tensor in (cu_seqlens_q, seqused_k, block_table)
    ):
        return False
    if k.shape[1] != 64 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    expected_inner_strides = (k.shape[-2] * k.shape[-1], k.shape[-1], 1)
    if k.stride()[1:] != expected_inner_strides:
        return False
    if v.stride()[1:] != expected_inner_strides:
        return False
    if min(k.stride(0), v.stride(0)) < 64 * expected_inner_strides[0]:
        return False
    if k.shape[-2] <= 0 or q.shape[-2] % k.shape[-2] != 0:
        return False
    # The vendor paged-attention kernel flattens the query-token and query-head
    # axes and accepts at most 64 such heads per KV head.  The qlen=7 route
    # below uses varlen_fwd instead and does not have this restriction.
    if causal and q.shape[-2] * query_len > k.shape[-2] * 64:
        return False
    max_seqlen_k = kwargs.get("max_seqlen_k")
    if (
        not isinstance(max_seqlen_k, int)
        or max_seqlen_k <= 0
        or max_seqlen_k > _DSPARK_MAX_CONTEXT_LENGTH
    ):
        return False
    if q.shape[0] > _DSPARK_MAX_CAPTURE_TOKENS:
        return False
    if block_table.shape[1] * 64 < max_seqlen_k:
        return False
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, out)):
        return False
    if not causal:
        scratch_bytes = (
            2
            * batch_size
            * max_seqlen_k
            * k.shape[-2]
            * k.shape[-1]
            * k.element_size()
        )
        if scratch_bytes > _DSPARK_MAX_STATIC_KV_SCRATCH_BYTES:
            return False
    if bool(kwargs.get("causal", False)) is not causal:
        return False
    if window_size is not None:
        if not isinstance(window_size, (list, tuple)):
            return False
        if len(window_size) != 2 or tuple(window_size) != (-1, -1):
            return False
    if kwargs.get("dropout_p", 0.0) != 0.0:
        return False
    if kwargs.get("softcap", 0.0) != 0.0:
        return False
    if kwargs.get("alibi_slopes") is not None or kwargs.get("s_aux") is not None:
        return False
    if kwargs.get("q_v") is not None:
        return False
    if kwargs.get("return_attn_probs", False):
        return False
    if kwargs.get("return_softmax_lse", False):
        return False
    if kwargs.get("cu_seqlens_k") is not None:
        return False
    if kwargs.get("deterministic", False):
        return False
    if kwargs.get("scheduler_metadata") is not None:
        return False
    if kwargs.get("num_splits", 0) != 0:
        return False
    if kwargs.get("cp_world_size", 1) != 1:
        return False
    if kwargs.get("cp_rank", 0) != 0 or kwargs.get("cp_tot_seqused_k") is not None:
        return False
    for name in ("q_descale", "k_descale", "v_descale"):
        scale = kwargs.get(name)
        if scale is not None and (
            not isinstance(scale, Tensor) or scale.device != q.device
        ):
            return False
    return True


@functools.wraps(_flash_attn_varlen_func)
def _flash_attn_varlen_func_with_dspark_capture(
    *args: Any,
    **kwargs: Any,
):
    if args:
        return _flash_attn_varlen_func(*args, **kwargs)

    use_paged_attention = _matches_dspark_attention_shape(
        kwargs,
        query_len=8,
        causal=True,
    )
    use_static_varlen = _matches_dspark_attention_shape(
        kwargs,
        query_len=7,
        causal=False,
    )
    if not use_paged_attention and not use_static_varlen:
        return _flash_attn_varlen_func(**kwargs)

    from flash_attn.flash_attn_interface import flash_attn_cuda

    q = kwargs["q"]
    k = kwargs["k"]
    v = kwargs["v"]
    out = kwargs["out"]
    cu_seqlens_q = kwargs["cu_seqlens_q"]
    batch_size = cu_seqlens_q.shape[0] - 1
    softmax_scale = kwargs.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if use_static_varlen:
        # DSpark's non-causal draft block cannot use paged_attention, which
        # applies a causal mask. Gather into a statically sized buffer instead
        # of materializing seqused_k.sum() on the host during graph capture.
        # The matcher bounds this scratch space to 584 MiB per rank, matching
        # Qwen3-8B TP2 at its maximum admitted shape:
        # 2 * floor(512 / 7) * 4096 * 4 * 128 * sizeof(bfloat16).
        max_seqlen_k = kwargs["max_seqlen_k"]
        total_k = batch_size * max_seqlen_k
        contiguous_k = torch.empty(
            (total_k, k.shape[-2], k.shape[-1]),
            device=k.device,
            dtype=k.dtype,
        )
        contiguous_v = torch.empty_like(contiguous_k)
        cu_seqlens_k = torch.empty_like(cu_seqlens_q)
        flash_attn_cuda.pagedkv_to_contiguouskv(
            contiguous_k,
            contiguous_v,
            k,
            v,
            kwargs["block_table"],
            kwargs["seqused_k"],
            cu_seqlens_k,
            max_seqlen_k,
            2,
        )
        flash_attn_cuda.varlen_fwd(
            q,
            contiguous_k,
            contiguous_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            None,
            7,
            max_seqlen_k,
            0.0,
            softmax_scale,
            False,
            False,
            -1,
            -1,
            0.0,
            False,
            kwargs.get("q_descale"),
            kwargs.get("k_descale"),
            kwargs.get("v_descale"),
            None,
            None,
        )
        return out

    flash_attn_cuda.paged_attention(
        out,
        q.reshape(batch_size, 8, q.shape[-2], q.shape[-1]),
        k,
        v,
        softmax_scale,
        kwargs["block_table"],
        kwargs["seqused_k"],
        None,
        "",
        kwargs.get("q_descale"),
        kwargs.get("k_descale"),
        kwargs.get("v_descale"),
        kwargs["max_seqlen_k"],
        None,
        2,
    )
    return out


flash_attn_varlen_func = _with_kv_cache_layout(
    _flash_attn_varlen_func_with_dspark_capture,
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

    The native HCU writer follows vLLM's stride-aware cache contract for FP8,
    avoiding AITER's incompatible FP8 conversion on HCU while supporting both
    NHD and HND storage. Existing non-quantized paths retain their optimized
    layout-specific writers.
    """
    if kv_cache_dtype in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        torch.ops.hcu_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    # Logical cache views remain [block, token, head, dim] in both layouts.
    # HND is distinguishable by a token stride smaller than the head stride.
    # If either dimension is one, selecting NHD is also address-equivalent.
    if key_cache.ndim == 4 and key_cache.stride(1) < key_cache.stride(2):
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    from aiter.ops.cache import (
        reshape_and_cache_flash as aiter_reshape_and_cache_flash,
    )

    aiter_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
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
