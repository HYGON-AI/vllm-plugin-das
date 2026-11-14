# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional, checked QK-FP8 dispatch for BF16 sparse MLA prefill inputs.

The native QK path does not implement padding/count masks or max_logits.
Keep the public frontend for every other case, including graph capture.
"""

import os
from contextlib import nullcontext

import torch

_dispatch_reported = False


def _supports_qk_fp8(q, kv, indices, d_v, attn_sink):
    if (
        "FLASH_MLA_DISABLE_DSA_MLS_PREFILL" in os.environ
        or attn_sink is not None
        or d_v != 512
        or not (q.is_cuda and kv.is_cuda and indices.is_cuda)
        or not (q.device == kv.device == indices.device)
        or not (q.dim() == kv.dim() == indices.dim() == 3)
        or q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or indices.dtype != torch.int32
        or q.shape[0] == 0
        or q.shape[-1] != 576
        or kv.shape[-1] != 576
        or kv.shape[1] != 1
        or q.shape[1] not in (64, 128)
        or tuple(indices.shape) != (q.shape[0], 1, 2048)
        or kv.shape[0] < 2048
        or any(t.stride(-1) != 1 for t in (q, kv, indices))
        or min(q.stride(0), q.stride(1), kv.stride(0), indices.stride(0)) <= 0
    ):
        return False
    arch = getattr(torch.cuda.get_device_properties(q.device), "gcnArchName", "")
    if "gfx938" not in arch:
        return False
    if kv.shape[0] >= 24576:
        return True
    return (
        q.shape[1] == 64
        and os.environ.get("FLASH_MLA_ENABLE_MLS_RESIDENT_PREFILL", "1") == "1"
        and kv.stride(0) < 8192
    )


class _UnsupportedQkFp8(ValueError):
    """An unsupported input, distinct from errors raised by native execution."""


def _prefer_qk_fp8(q, kv, with_sink=False):
    # Native QK-FP8 quantizes the full KV workspace, even though attention
    # gathers only 2048 entries per query. Avoid paying that fixed cost when
    # query/head work is too small to amortize it (gfx938 operator sweep).
    if q.ndim != 3 or kv.ndim != 3:
        return True  # Let the ordinary shape checks/fallback handle this input.
    work = q.shape[0] * q.shape[1]
    if with_sink:
        # Sink correction adds an output pass. Keep the marginal workloads on
        # BF16; this conservative bound includes the long-workspace sweep.
        return work > 32768 and 8 * work > kv.shape[0] + 131072
    # Bound the fallback to small query/head workloads: a KV/work ratio alone
    # incorrectly rejects larger workloads that already amortize preprocessing.
    return work >= 32768 or 2 * work > kv.shape[0]


def _has_invalid_rows(indices, topk_length, kv_length):
    if indices.is_cuda:
        from vllm_hcu.v1.attention.ops.flashmla_qk_fp8_validation import invalid_rows

        return invalid_rows(indices, topk_length, kv_length).any().item()
    # CPU reference for the dispatch contract tests; production is CUDA-only.
    invalid = ((indices < 0) | (indices >= kv_length)).any()
    if topk_length is not None:
        invalid = invalid | (topk_length != 2048).any()
    return invalid.item()


def _supports_sink(q, sink):
    return (
        isinstance(sink, torch.Tensor)
        and q.ndim == 3
        and sink.dtype == torch.float32
        and sink.shape == (q.shape[1],)
        and sink.device == q.device
        and sink.stride(0) > 0
    )


def _apply_sink(output, lse, sink):
    from vllm_hcu.v1.attention.ops.flashmla_qk_fp8_sink import apply_sink

    apply_sink(output, lse, sink)
    return output


def _report_dispatch(q, kv, has_sink):
    global _dispatch_reported
    if q.is_cuda and not _dispatch_reported:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "FlashMLA QK-FP8 sparse prefill selected: Q=%s, KV=%s, sink=%s",
            tuple(q.shape), tuple(kv.shape), has_sink,
        )
        _dispatch_reported = True


def sparse_prefill_qk_fp8(q, kv, indices, sm_scale, d_v=512, *, topk_length=None):
    """The documented direct QK-FP8 entry, returning only output and LSE.

    vLLM may supply counts; they can be omitted at the native boundary only
    after proving every count is 2048. This eager check synchronizes once.
    """
    if os.environ.get("FLASH_MLA_DSA_FP8_QK") != "1":
        raise _UnsupportedQkFp8("FLASH_MLA_DSA_FP8_QK must be '1'")
    if not _supports_qk_fp8(q, kv, indices, d_v, None):
        raise _UnsupportedQkFp8("unsupported QK-FP8 shape, dtype, device or layout")
    if topk_length is not None and (
        topk_length.shape != (q.shape[0],)
        or topk_length.dtype != torch.int32
        or topk_length.device != indices.device
    ):
        raise _UnsupportedQkFp8("unsupported topk_length metadata")
    context = torch.cuda.device(q.device) if q.is_cuda else nullcontext()
    with context:
        if torch.cuda.is_current_stream_capturing():
            raise _UnsupportedQkFp8("QK-FP8 validation requires eager execution")
        if _has_invalid_rows(indices, topk_length, kv.shape[0]):
            raise _UnsupportedQkFp8(
                "indices must be in range and topk_length must be 2048"
            )
        from flash_mla.flash_mla_interface import flash_mla_cuda

        output, _unused_max, lse = flash_mla_cuda.sparse_prefill_fwd(
            q, kv, indices, sm_scale, d_v, None, None
        )
        return output, lse


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd as public

    if os.environ.get("FLASH_MLA_DSA_FP8_QK") != "1":
        return public(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
    if _prefer_qk_fp8(q, kv, attn_sink is not None):
        if attn_sink is not None and not _supports_sink(q, attn_sink):
            return public(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
        try:
            output, lse = sparse_prefill_qk_fp8(
                q, kv, indices, sm_scale, d_v, topk_length=topk_length
            )
            if attn_sink is not None:
                output = _apply_sink(output, lse, attn_sink)
            _report_dispatch(q, kv, attn_sink is not None)
            # max_logits is uninitialized in the QK kernel. Existing MLA callers
            # consume output/LSE only; never expose its uninitialized buffer.
            return output, None, lse
        except _UnsupportedQkFp8:
            pass
    if topk_length is None:
        # A non-None count prevents the public frontend selecting QK-FP8
        # when shape cost, padding or graph capture forced fallback.
        topk_length = torch.full(
            (q.shape[0],),
            indices.shape[-1],
            dtype=torch.int32,
            device=indices.device,
        )
    return public(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
