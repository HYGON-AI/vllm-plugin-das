# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Default-on FlashMLA prefill for V4.1's gathered BF16 KV workspace.

Keep index selection, cache gathering, and inverse-RoPE in the model. Only
replace the attention computation, and never change the decode implementation.
The initial contract is limited to the gfx936/gfx938 D512 configurations probed with
the native FlashMLA library. Unsupported inputs retain the original operator.
"""

from __future__ import annotations

import functools
import logging

import torch

from vllm_hcu.platforms import envs as henvs

logger = logging.getLogger(__name__)
_SUPPORTED_HEADS = frozenset({4, 8, 16, 64, 128})
_SUPPORTED_ARCHES = frozenset({"gfx936", "gfx938"})


def _supported_device(q: torch.Tensor) -> bool:
    if q.device.type != "cuda" or torch.version.hip is None:
        return False
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
    return arch in _SUPPORTED_ARCHES


@functools.cache
def _flashmla_prefill():
    # Lazy: disabling the experiment must not import FlashMLA or initialize HIP.
    try:
        from vllm_hcu.v1.attention.ops.flashmla import (
            flash_mla_sparse_fwd,
            is_flashmla_sparse_supported,
        )
    except ImportError:
        return None
    if not is_flashmla_sparse_supported()[0]:
        return None
    return flash_mla_sparse_fwd


@functools.cache
def _log_fallback(reason: str) -> None:
    logger.info("V4.1 FlashMLA prefill: using original operator (%s)", reason)


@functools.cache
def _log_selected() -> None:
    logger.info("V4.1 prefill: using native FlashMLA sparse attention (decode unchanged)")


def try_flashmla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> bool:
    """Write ``output`` and return True, or return False without touching it.

    Do not catch kernel errors: after a HIP launch it is unsafe to silently run
    another kernel, and doing so would hide correctness failures in this trial.
    All dispatch checks use tensor metadata, without device-to-host syncs.
    """
    if not henvs.dsv41_flashmla_prefill_enabled():
        return False
    if ragged_indices is not None or ragged_indptr is not None:
        _log_fallback("ragged input")
        return False
    if (head_dim, nope_head_dim, rope_head_dim) != (512, 448, 64):
        _log_fallback("only D512 / NoPE448 / RoPE64 is validated")
        return False
    if (
        q.ndim != 3
        or kv.ndim != 3
        or kv.shape[1:] != (1, 512)
        or q.shape[-1] != 512
        or q.shape[0] == 0
        or kv.shape[0] == 0
        or q.shape[1] not in _SUPPORTED_HEADS
        or output.shape != q.shape
        or q.dtype != torch.bfloat16
        or kv.dtype != q.dtype
        or output.dtype != q.dtype
        or kv.device != q.device
        or output.device != q.device
        or q.stride(-1) != 1
        or kv.stride(-1) != 1
    ):
        _log_fallback("unsupported query/KV/output shape, dtype, or stride")
        return False
    if (
        indices is None
        or indices.ndim not in (2, 3)
        or indices.shape[0] != q.shape[0]
        or (indices.ndim == 3 and indices.shape[1] != 1)
        or indices.shape[-1] == 0
        or indices.shape[-1] % 128 != 0
        or indices.dtype != torch.int32
        or indices.device != q.device
    ):
        _log_fallback("dense int32 indices must have a 128-aligned width")
        return False
    if topk_length is not None and (
        topk_length.shape != (q.shape[0],)
        or topk_length.dtype != torch.int32
        or topk_length.device != q.device
    ):
        _log_fallback("unsupported topk_length")
        return False
    if attn_sink is not None and (
        attn_sink.ndim != 1
        or attn_sink.numel() < q.shape[1]
        or attn_sink.dtype != torch.float32
        or attn_sink.device != q.device
    ):
        _log_fallback("unsupported attention sink")
        return False
    if not _supported_device(q):
        _log_fallback("only gfx936/gfx938 are validated")
        return False
    kernel = _flashmla_prefill()
    if kernel is None:
        _log_fallback("native FlashMLA unavailable")
        return False

    # Preserve caller-owned indices even if the native wrapper sorts in-place.
    # Mask inactive columns explicitly: FlashMLA documents a NaN hazard when a
    # valid index beyond topk_length points into an uninitialized gathered slot.
    dense_indices = indices.reshape(q.shape[0], 1, -1)
    if topk_length is None:
        dense_indices = dense_indices.clone()
    else:
        columns = torch.arange(dense_indices.shape[-1], device=q.device)
        dense_indices = torch.where(
            columns[None, None, :] < topk_length[:, None, None], dense_indices, -1
        )
    sink = None if attn_sink is None else attn_sink[: q.shape[1]].contiguous()
    result, _, _ = kernel(
        q=q,
        kv=kv,
        indices=dense_indices,
        sm_scale=scale,
        d_v=512,
        attn_sink=sink,
        topk_length=None if topk_length is None else topk_length.contiguous(),
    )
    output.copy_(result)
    _log_selected()
    return True
