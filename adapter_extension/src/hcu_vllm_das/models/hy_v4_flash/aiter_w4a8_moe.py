# SPDX-License-Identifier: Apache-2.0
"""Explicit Flash AITER channel INT4/token INT8 candidate.

CT owns parameter creation/loading and completeness checks. AITER owns native
layout preparation, routing and arithmetic. Its activation rounding differs
from the exact CT reference; full-model qualification is required separately.
"""
from types import SimpleNamespace
import math

import torch

from .w4a8 import (
    HYV4FlashChannelW4A8Method,
    HYV4FlashW4A8Config,
    packed_byte_view,
)


def ct_to_aiter_packed(value):
    """Losslessly convert CT offset/low-even nibbles to HIPC signed/high-even."""
    if value.dtype != torch.uint8 or value.ndim != 3 or not value.is_contiguous():
        raise ValueError('AITER packing needs contiguous uint8 [experts, N, K/2]')
    return (((value << 4) | (value >> 4)) ^ 0x88).contiguous()


def _device_supported(value):
    if value.device.type != 'cuda':
        return False
    return torch.cuda.get_device_properties(value.device).gcnArchName.split(':')[0] == 'gfx938'


def _vendor_api():
    # Explicit imports avoid treating the top-level fused_moe module as a
    # function. Missing requested providers and real kernel errors propagate.
    from aiter import QuantType
    from aiter.fused_moe import fused_moe
    from aiter.fused_moe_layout import fused_moe_prepare_weights
    from aiter.moe import MoeQuantType, get_aiter_moe_config
    from hcu_adapter._base import Missing

    for value in (fused_moe, fused_moe_prepare_weights, get_aiter_moe_config):
        if isinstance(value, Missing) or not callable(value):
            raise RuntimeError('Requested AITER W4A8 public API is unavailable')
    return SimpleNamespace(execute=fused_moe, prepare=fused_moe_prepare_weights,
                           config=get_aiter_moe_config, quant_type=QuantType.per_Token,
                           w4a8=MoeQuantType.W4A8)


def _identity(layer):
    return tuple((id(p), p.data_ptr(), tuple(p.shape), tuple(p.stride()), p.dtype, p.device)
                 for p in (layer.w13_weight_packed, layer.w2_weight_packed,
                           layer.w13_weight_scale, layer.w2_weight_scale))


def _unique_padded_route_ids(ids):
    """Fill padding with distinct unused IDs, retaining Core's valid routes.

    The native sorter has one mesh cell per (expert, token), so repeated IDs
    lose route slots even when their weights are zero. Core's valid top-k IDs
    are distinct. Among K candidate IDs there are always enough unused IDs for
    K minus the valid-slot count. All masks/ranks stay on the device.
    """
    valid = ids >= 0
    candidates = torch.arange(ids.shape[1], device=ids.device, dtype=ids.dtype)
    blocked = (ids[:, :, None] == candidates).any(dim=1)
    free_rank = (~blocked).to(torch.int32).cumsum(-1) - 1
    pad_rank = (~valid).to(torch.int32).cumsum(-1) - 1
    match = (free_rank[:, None, :] == pad_rank[:, :, None]) & (~blocked[:, None, :])
    dummy = (match.to(ids.dtype) * candidates).sum(-1).to(ids.dtype)
    return torch.where(valid, ids, dummy).to(torch.int32)


class HYV4FlashAiterW4A8Config(HYV4FlashW4A8Config):
    """Inherit serialized CT admission, overriding only admitted routed experts."""

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, HYV4FlashChannelW4A8Method):
            if layer.moe_config.moe_backend != 'aiter':
                raise ValueError('Flash AITER W4A8 requires explicit moe_backend=aiter')
            return HYV4FlashAiterW4A8Method(layer.moe_config)
        return method


class HYV4FlashAiterW4A8Method(HYV4FlashChannelW4A8Method):
    """Use the exact CT loader ABI and a single postload native packed owner."""

    def create_weights(self, layer, *args, **kwargs):
        if self.moe.moe_backend != 'aiter':
            raise ValueError('Flash AITER W4A8 requires explicit moe_backend=aiter')
        return super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        if getattr(self, '_prepared', None) is not None:
            raise RuntimeError('AITER W4A8 weights already processed; cannot repack')
        if self.moe.moe_backend != 'aiter':
            raise ValueError('Flash AITER W4A8 requires explicit moe_backend=aiter')
        if not _device_supported(layer.w13_weight_packed):
            raise NotImplementedError('Flash AITER W4A8 is qualified only for gfx938')
        super().process_weights_after_loading(layer)
        self._weights_processed = False
        api = _vendor_api()
        limit = layer.swiglu_limit
        if limit is not None and (type(limit) not in (float, int) or not math.isfinite(limit) or limit <= 0):
            raise ValueError('AITER W4A8 requires a positive SwiGLU clamp or None')
        experts = layer.w13_weight_packed.shape[0]
        if not 0 < self.moe.experts_per_token <= experts:
            raise ValueError('AITER W4A8 requires 0 < topk <= number of experts')
        # The config query is retained as vendor evidence. Preparation is the
        # public fused API's authoritative layout decision over actual capacity.
        ok, self._vendor_config = api.config(
            1, experts, 2 * self.intermediate, self.hidden, self.hidden,
            self.moe.experts_per_token, 0, torch.bfloat16, api.w4a8,
            activation='silu', gated=True,
        )
        if not ok:
            raise RuntimeError('AITER has no W4A8 configuration for these experts')
        w1 = ct_to_aiter_packed(packed_byte_view(layer.w13_weight_packed))
        w2 = ct_to_aiter_packed(packed_byte_view(layer.w2_weight_packed))
        w1, w2, prepared = api.prepare(
            w1, w2, quant_type=api.quant_type,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            dtype=torch.bfloat16, topk=self.moe.experts_per_token,
            activation='silu', quant_dtype_a=torch.int8, backend='auto',
            max_num_tokens=self.moe.max_num_tokens,
        )
        if any(w.dtype not in (torch.uint8, torch.int32, torch.uint32) for w in (w1, w2)):
            raise RuntimeError('AITER W4A8 preparation must retain packed integer weights')
        # Keep Parameter objects/loader attributes, replacing their CT storage
        # once before Graph capture. Prepared views share native storage; the
        # old full-model CT packed storage is not retained as another copy.
        # Preserve CT's INT32 Parameter dtype and element-count ABI as well as
        # its Python owner; the uint8 native views alias these same bytes.
        owned1 = w1.view(layer.w13_weight_packed.dtype)
        owned2 = w2.view(layer.w2_weight_packed.dtype)
        layer.w13_weight_packed.data = owned1
        layer.w2_weight_packed.data = owned2
        layer.w13_weight_scale.data = prepared.w1_scale
        layer.w2_weight_scale.data = prepared.w2_scale
        self._native_weights = (w1, w2)
        self._prepared = prepared
        self._api = api
        self._limit = limit
        self._lifecycle = _identity(layer)
        self._weights_processed = True

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if not self._weights_processed or getattr(self, '_prepared', None) is None:
            raise RuntimeError('AITER W4A8 weights must be processed after CT loading')
        if _identity(layer) != self._lifecycle or layer.swiglu_limit != self._limit:
            raise RuntimeError('AITER W4A8 storage/lifecycle changed; rebuild the method')
        if x.dtype != torch.bfloat16 or not _device_supported(x):
            raise ValueError('AITER W4A8 requires gfx938 BF16 activations')
        if (x.ndim != 2 or x.shape[1] != self.hidden or x.shape[0] > self.moe.max_num_tokens
                or topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2
                or topk_ids.shape != (x.shape[0], self.moe.experts_per_token)
                or topk_ids.dtype not in (torch.int32, torch.int64)
                or topk_weights.dtype != torch.float32
                or topk_ids.device != x.device or topk_weights.device != x.device):
            raise ValueError('Invalid AITER W4A8 activation/router shape, dtype or capacity')
        if torch.cuda.is_current_stream_capturing() and not self._graph_admitted:
            raise RuntimeError('Eager AITER W4A8 configuration cannot capture a Graph')
        if x.shape[0] == 0:
            return torch.empty_like(x)
        experts = layer.w13_weight_packed.shape[0]
        torch._assert_async(torch.all((topk_ids >= -1) & (topk_ids < experts)),
                            'AITER W4A8 expert ID out of range')
        valid = topk_ids >= 0
        active = (valid & (topk_weights != 0)).any(dim=1, keepdim=True)
        # Core Graph padding may contain -1 routes, NaN weights/activations.
        # Mask both operands before multiplication and keep all padding on GPU.
        values = torch.where(active, x, 0)
        weights = torch.where(valid, topk_weights, 0)
        ids = _unique_padded_route_ids(topk_ids)
        result = self._api.execute(
            values, *self._native_weights, weights, ids,
            quant_type=self._api.quant_type, quant_dtype_a=torch.int8,
            gemm1_limit=self._limit, prepared=self._prepared,
        )
        return torch.where(active, result, 0)
