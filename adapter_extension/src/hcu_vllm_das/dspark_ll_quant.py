# SPDX-License-Identifier: Apache-2.0
"""Quantize draft LL receive buffers with the same public op as AG/RS."""

import hashlib
import logging
import os
from functools import lru_cache
from inspect import getsource, signature

import torch
from vllm.model_executor.layers.quantization.utils import int8_utils as native
from vllm.triton_utils import tl, triton

_LOG = logging.getLogger(__name__)
_NATIVE_ROW = getattr(native, '_per_token_quant_int8', None)
_LIVE_ROWS_ENV = 'HCU_DSV41_DRAFT_LL_LIVE_QUANT'
# Qualify only the three functions that determine the row arithmetic and launch.
_REVIEWED = (
    'cd0c98032382a9d13914f6dcf39360541f90f6f203fa102fa0442e85e17d3544',
    'e54e5c4c0d9906185bc279a18270f99232a3010a4b1e72eff2c524c3b2ec3059',
    'c5e1233365d0c1f48be6195f8900473d738e9ed2ee81aedb56c18eb45301a72e',
)


@triton.jit
def _quantize_draft_ll_live_rows(X, Q, S, COUNTS, CAPACITY: tl.constexpr,
                                 ROW_STRIDE: tl.constexpr):
    slot = tl.program_id(0)
    expert = tl.program_id(1)
    row = slot
    end = tl.minimum(tl.maximum(tl.load(COUNTS + expert), 0), CAPACITY)
    while row < end:
        # The native body adds program_id(0) to its row address.
        offset = expert * CAPACITY + row - slot
        _NATIVE_ROW(X + offset * 5120, Q + offset * 5120, S + offset,
                    5120, 5120, 5120, BLOCK=8192)
        row += ROW_STRIDE


@lru_cache(maxsize=None)
def _live_rows_enabled(device):
    if os.environ.get(_LIVE_ROWS_ENV) != '1' or not torch.version.hip:
        return False
    props = torch.cuda.get_device_properties(device)
    if getattr(props, 'gcnArchName', '').split(':')[0] != 'gfx938':
        _LOG.info('HCU draft LL live-row INT8 uses full-row fallback on %s', props.name)
        return False
    public = _receiver_quantizer()
    if (public is None or public is not getattr(native, 'per_token_quant_int8', None)
            or _NATIVE_ROW is not getattr(native, '_per_token_quant_int8', None)):
        _LOG.info('HCU draft LL live-row INT8 uses full-row fallback for changed entry points')
        return False
    providers = (public, _NATIVE_ROW, getattr(native, 'round_int8', None))
    try:
        signatures = tuple(hashlib.sha256(getsource(getattr(fn, 'fn', fn)).encode()).hexdigest()
                           for fn in providers)
    except (OSError, TypeError):
        signatures = ()
    if signatures != _REVIEWED:
        _LOG.info('HCU draft LL live-row INT8 uses full-row fallback for changed row semantics')
        return False
    _LOG.info('HCU draft LL live-row INT8 admitted on %s', props.gcnArchName)
    return True


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

    Both paths use the AG/RS row arithmetic. The opt-in path uses the current
    device counts and leaves padding unspecified for the masked GEMM.
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
    if x.shape[0] == 16 and x.shape[1] <= 2048 and _live_rows_enabled(x.device):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        row_stride = min(32, x.shape[1])
        _quantize_draft_ll_live_rows[(row_stride, x.shape[0])](
            x, q, scales, counts, x.shape[1], row_stride, num_warps=8, num_stages=1)
        return q, scales
    q, scales = quantize(x.view(-1, x.shape[-1]))
    if (q.dtype != torch.int8 or q.device != x.device
            or q.shape != (x.shape[0] * x.shape[1], x.shape[2])
            or scales.dtype != torch.float32 or scales.device != x.device
            or scales.shape != (q.shape[0], 1)
            or not q.is_contiguous() or not scales.is_contiguous()):
        raise RuntimeError('W4A8 draft LL public quantizer returned an incompatible layout')
    return q.view_as(x), scales.view(*x.shape[:2], 1)
