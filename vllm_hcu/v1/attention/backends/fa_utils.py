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
from vllm.triton_utils import tl, triton
from vllm_hcu.v1.attention.kv_cache_layout import get_kv_cache_layout

import vllm_hcu.hcu_ops as hcu_ops


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


@triton.jit
def _gather_paged_kv_kernel(
    key_cache,
    value_cache,
    key_out,
    value_out,
    block_table,
    seq_lens,
    cu_seqlens,
    block_table_stride,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    page_size: tl.constexpr,
    block_size: tl.constexpr,
):
    token_offset = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads
    columns = tl.arange(0, block_size)
    token_valid = token_offset < tl.load(seq_lens + batch_idx)
    valid = token_valid & (columns < head_size)
    block_id = tl.load(
        block_table
        + batch_idx * block_table_stride
        + token_offset // page_size,
        mask=token_valid,
        other=0,
    ).to(tl.int64)
    page_offset = token_offset % page_size
    output_token = tl.load(cu_seqlens + batch_idx) + token_offset
    key_offsets = (
        block_id * key_stride_0
        + page_offset * key_stride_1
        + head_idx * key_stride_2
        + columns * key_stride_3
    )
    value_offsets = (
        block_id * value_stride_0
        + page_offset * value_stride_1
        + head_idx * value_stride_2
        + columns * value_stride_3
    )
    output_offsets = (
        output_token * out_stride_0
        + head_idx * out_stride_1
        + columns * out_stride_2
    )
    tl.store(key_out + output_offsets, tl.load(key_cache + key_offsets, mask=valid), mask=valid)
    tl.store(
        value_out + output_offsets,
        tl.load(value_cache + value_offsets, mask=valid),
        mask=valid,
    )


def _gather_paged_kv(
    key_cache: Tensor,
    value_cache: Tensor,
    key_out: Tensor,
    value_out: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> None:
    num_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]
    _gather_paged_kv_kernel[(max_seqlen, seq_lens.numel() * num_heads)](
        key_cache,
        value_cache,
        key_out,
        value_out,
        block_table,
        seq_lens,
        cu_seqlens,
        block_table.stride(0),
        *key_cache.stride(),
        *value_cache.stride(),
        *key_out.stride(),
        num_heads=num_heads,
        head_size=head_size,
        page_size=key_cache.shape[1],
        block_size=triton.next_power_of_2(head_size),
    )


@functools.wraps(_flash_attn_varlen_func)
def _safe_flash_attn_varlen_func(*args: Any, **kwargs: Any):
    """Avoid an undersized vendor gather buffer for long chunked prefill."""
    query = kwargs.get("q")
    key_cache = kwargs.get("k")
    value_cache = kwargs.get("v")
    block_table = kwargs.get("block_table")
    seq_lens = kwargs.get("seqused_k")
    max_seqlen_q = kwargs.get("max_seqlen_q", 0)
    max_seqlen_k = kwargs.get("max_seqlen_k", 0)

    if (
        not isinstance(query, torch.Tensor)
        or not isinstance(key_cache, torch.Tensor)
        or not isinstance(value_cache, torch.Tensor)
        or not isinstance(block_table, torch.Tensor)
        or not isinstance(seq_lens, torch.Tensor)
        or query.shape[0] > 256
        or int(max_seqlen_q) <= 4
        or int(max_seqlen_k) < 8192
        or int(max_seqlen_k) <= int(max_seqlen_q)
    ):
        return _flash_attn_varlen_func(*args, **kwargs)

    layout = kwargs.get("layout", "bshd")
    if layout == "bhsd":
        block_size = key_cache.shape[-2]
        token_major_key = key_cache.transpose(1, 2)
        token_major_value = value_cache.transpose(1, 2)
    elif layout == "bshd":
        block_size = key_cache.shape[1]
        token_major_key = key_cache
        token_major_value = value_cache
    else:
        return _flash_attn_varlen_func(*args, **kwargs)

    key_size = token_major_key.shape[-1]
    value_size = token_major_value.shape[-1]
    if (
        block_size != 64
        or key_size != value_size
        or token_major_key.shape[:-1] != token_major_value.shape[:-1]
    ):
        return _flash_attn_varlen_func(*args, **kwargs)

    kv_dtype = key_cache.dtype
    if key_cache.dtype == torch.float8_e5m2:
        output = kwargs.get("out")
        if isinstance(output, torch.Tensor):
            kv_dtype = output.dtype
        elif query.dtype == torch.float8_e5m2:
            kv_dtype = torch.bfloat16
        else:
            kv_dtype = query.dtype
    query = query.to(kv_dtype)

    total_k = int(seq_lens.sum().item())
    contiguous_key = torch.empty(
        (total_k, token_major_key.shape[-2], key_size),
        device=query.device,
        dtype=kv_dtype,
    )
    contiguous_value = torch.empty_like(contiguous_key)
    cu_seqlens_k = torch.zeros_like(kwargs["cu_seqlens_q"])
    torch.cumsum(seq_lens, dim=0, out=cu_seqlens_k[1:])
    _gather_paged_kv(
        token_major_key,
        token_major_value,
        contiguous_key,
        contiguous_value,
        block_table,
        seq_lens,
        cu_seqlens_k,
        int(max_seqlen_k),
    )

    kwargs.update(
        q=query,
        k=contiguous_key,
        v=contiguous_value,
        cu_seqlens_k=cu_seqlens_k,
        block_table=None,
        seqused_k=None,
    )
    if kwargs.get("return_softmax_lse"):
        kwargs["return_attn_probs"] = True
    return _flash_attn_varlen_func(*args, **kwargs)


flash_attn_varlen_func = _with_kv_cache_layout(
    _safe_flash_attn_varlen_func,
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
