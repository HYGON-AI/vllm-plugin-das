# SPDX-License-Identifier: Apache-2.0
"""CT packed weights, HCU HIPC representation and scale domain.

The generic half — creating the packed parameters and letting the native
RoutedExperts loader fill its shards — lives in
`hcu_adapter.ct_w4a8_packed_loader` and is subclassed here. What stays is what
is genuinely private to the HCU HIPC representation: the nibble reorder with the
XOR mask, the /16 channel-scale domain with its guard allocation, and the N32
view shape. No backend or selector is published here.
"""
from __future__ import annotations

import torch

from hcu_adapter.ct_w4a8_packed_loader import PackedCTParameters
from vllm.model_executor.utils import replace_parameter

_GUARD_FLOATS = (2 * 1024 * 1024) // 4


def convert_ct_int4_to_hipc_bytes_(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32 or packed.ndim != 3 or not packed.is_contiguous():
        raise ValueError("CT W4A8 requires contiguous rank-3 packed INT32 weights")
    data = packed.view(torch.uint8)
    high = data >> 4
    data.bitwise_left_shift_(4).bitwise_or_(high).bitwise_xor_(0x88)
    return data.view(torch.int8)


def hipc_channel_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.float32 or scale.ndim != 3 or scale.shape[-1] != 1:
        raise ValueError("CT W4A8 channel scales must be FP32 [E,N,1]")
    if not scale.is_contiguous() or scale.numel() == 0:
        raise ValueError("CT W4A8 channel scales must be nonempty and contiguous")
    storage = torch.empty(scale.numel() + _GUARD_FLOATS, dtype=torch.float32, device=scale.device)
    result = storage[:scale.numel()].view(scale.shape)
    result.copy_(scale).div_(16)
    return result


class PackedCTWeights(PackedCTParameters):
    """Own raw CT parameters and a single transition to checked HIPC storage."""

    def pack(self, pack_weight, n32_view=None):
        if self.state == "ready":
            if self.signature() != self._signature_ready:
                raise RuntimeError("CT W4A8 parameters changed after packing; restart worker")
            return
        if self.state != "loaded":
            raise RuntimeError("CT W4A8 packing failed or is incomplete; restart worker")
        if not callable(pack_weight) or (n32_view is not None and not callable(n32_view)):
            raise ValueError("CT W4A8 requires callable HIPC pack/view providers")
        e, h, i = self.shape
        names = ("w13_weight_packed", "w2_weight_packed")
        device = self.layer.w13_weight_packed.device
        for name, shape in zip(names, ((e, 2*i, h//8), (e, h, i//8))):
            value = getattr(self.layer, name)
            if value.dtype != torch.int32 or tuple(value.shape) != shape or not value.is_contiguous() or value.device != device:
                raise ValueError(f"CT W4A8 invalid packed parameter {name}")
        scales = (self.layer.w13_weight_scale, self.layer.w2_weight_scale)
        for value, shape in zip(scales, ((e, 2*i, 1), (e, h, 1))):
            if value.dtype != torch.float32 or tuple(value.shape) != shape or value.device != device or not value.is_contiguous():
                raise ValueError("CT W4A8 invalid channel scale shape/dtype/device")
        self.state = "packing"
        try:
            self.scales = tuple(hipc_channel_scale(s) for s in scales)
            views = []
            for name in names:
                raw = getattr(self.layer, name)
                with torch.no_grad():
                    converted = convert_ct_int4_to_hipc_bytes_(raw)
                    packed = pack_weight(converted)
                if packed.dtype != torch.int8 or packed.shape != converted.shape or packed.data_ptr() != converted.data_ptr():
                    raise RuntimeError("HIPC packer must preserve packed INT8 storage and shape")
                replace_parameter(self.layer, name, packed)
                view = packed if n32_view is None else n32_view(packed)
                expected = (packed.shape[0], packed.shape[2]//32, packed.shape[1]//32, 4, 32, 8)
                if view.dtype != torch.int8 or view.data_ptr() != packed.data_ptr() or view.untyped_storage().data_ptr() != packed.untyped_storage().data_ptr():
                    raise RuntimeError("HIPC N32 view must alias packed weight storage")
                if n32_view is not None and tuple(view.shape) != expected:
                    raise RuntimeError("HIPC N32 view shape differs from the packed matrix")
                views.append(view)
            self.weights = tuple(views)
            self._signature_ready = self.signature()
            self.state = "ready"
        except BaseException:
            self.state = "failed"
            raise
