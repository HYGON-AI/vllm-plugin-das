# SPDX-License-Identifier: Apache-2.0
"""Quantize draft LL receive buffers with the same public op as AG/RS."""

from functools import lru_cache
from inspect import signature

import torch


@lru_cache(maxsize=1)
def _receiver_quantizer():
    from vllm.model_executor.layers.fused_moe import utils

    quantize = getattr(utils, 'per_token_quant_int8', None)
    if not callable(quantize):
        return None
    try:
        signature(quantize).bind(None)
    except (TypeError, ValueError):
        return None
    return quantize


def receiver_quantizer_available():
    return _receiver_quantizer() is not None


def quantize_receiver(x, counts):
    """Return native per-token INT8 and FP32 scales in masked GEMM layout.

    The public quantizer processes each row independently. Padding can be
    quantized too because its output is never consumed by the masked GEMM.
    """
    if (x.dtype != torch.bfloat16 or x.ndim != 3
            or x.shape[0] not in (16, 48) or x.shape[1] <= 0 or x.shape[2] != 5120
            or x.shape[1] % 8 != 0 or not x.is_cuda or not x.is_contiguous()
            or counts.dtype != torch.int32 or counts.shape != x.shape[:1]
            or counts.device != x.device or not counts.is_contiguous()):
        raise ValueError('W4A8 draft receiver requires the contiguous DP8 LL layout')
    quantize = _receiver_quantizer()
    if quantize is None:
        raise RuntimeError('W4A8 draft LL receiver selected without a quantizer')
    q, scales = quantize(x.view(-1, x.shape[-1]))
    if (q.dtype != torch.int8 or q.device != x.device
            or q.shape != (x.shape[0] * x.shape[1], x.shape[2])
            or scales.dtype != torch.float32 or scales.device != x.device
            or scales.shape != (q.shape[0], 1)
            or not q.is_contiguous() or not scales.is_contiguous()):
        raise RuntimeError('W4A8 draft LL public quantizer returned an incompatible layout')
    return q.view_as(x), scales.view(*x.shape[:2], 1)
