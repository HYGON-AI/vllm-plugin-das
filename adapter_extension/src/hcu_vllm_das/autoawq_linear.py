# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Optional LightOp layout adapter; loading and fallback remain official."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase

_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_LAYOUT_ATTR = "_vllm_hcu_lightop_autoawq_layout"
_DELEGATE_LAYOUT = "delegate"
_LIGHTOP_LAYOUT = "lightop"
_LIGHTOP_TUNED_KN = frozenset(
    {
        (8192, 10240),
        (8192, 8192),
        (8192, 59392),
        (29696, 8192),
        (8192, 5120),
        (4096, 8192),
        (8192, 29696),
        (14848, 8192),
        (8192, 2560),
        (2048, 8192),
        (8192, 14848),
        (7424, 8192),
        (8192, 1280),
        (1024, 8192),
        (8192, 7424),
        (3712, 8192),
    }
)


def is_lightop_awq_shape_supported(k: int, n: int) -> bool:
    return (k, n) in _LIGHTOP_TUNED_KN


def convert_awq_to_lightop_layout(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert standard AutoAWQ tensors to LightOp's logical input layout."""

    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AutoAWQ qweight, qzeros, and scales must be rank two")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AutoAWQ packed weights and zero points must be int32")
    if group_size <= 0:
        raise ValueError("AutoAWQ group size must be positive")

    k, n_packed = qweight.shape
    n = n_packed * 8
    if k % 8 != 0 or k % group_size != 0:
        raise ValueError("AutoAWQ K must be divisible by 8 and group size")
    expected_groups = k // group_size
    if qzeros.shape != (expected_groups, n_packed):
        raise ValueError("AutoAWQ qzeros shape does not match qweight")
    if scales.shape != (expected_groups, n):
        raise ValueError("AutoAWQ scales shape does not match qweight")

    shifts = (
        torch.tensor(
            _AWQ_PACK_ORDER,
            dtype=torch.int32,
            device=qweight.device,
        )
        * 4
    )
    logical_weight = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(k, n)
    weight_trans = torch.sum(
        (logical_weight.T.reshape(n, k // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()

    logical_zeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(expected_groups, n)
    zero_scale_pairs = torch.stack(
        (
            (logical_zeros + 64).to(torch.float16),
            scales.to(torch.float16),
        ),
        dim=-1,
    )
    scales_zeros = (
        zero_scale_pairs.contiguous().view(torch.int32).squeeze(-1).T.contiguous()
    )
    return weight_trans, scales_zeros


def _resolve_lightop_awq_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from lightop.gemm_ops import (
        awq_gemm_marlin_weight_repack,
        gemm_awq_w4a16_marlin,
    )

    expected_signatures = (
        (awq_gemm_marlin_weight_repack, ("weight_trans", "N", "K")),
        (gemm_awq_w4a16_marlin, ("a", "b", "scales_zeros")),
    )
    for function, expected in expected_signatures:
        try:
            actual = tuple(inspect.signature(function).parameters)
        except (TypeError, ValueError) as error:
            raise ImportError(
                "LightOp AutoAWQ public callable has no inspectable signature"
            ) from error
        if actual != expected:
            raise ImportError(
                "LightOp AutoAWQ public callable has incompatible signature: "
                f"expected {expected}, got {actual}"
            )
    return awq_gemm_marlin_weight_repack, gemm_awq_w4a16_marlin


def _matches_prefill_layout(
    logical_weight: torch.Tensor, packed_weight: torch.Tensor, k: int, n: int
) -> bool:
    """Check actual packed bytes before allowing a separate layout reader.

    The public vendor GEMM remains usable when a compatible repacker uses a
    different layout. This comparison adds no persistent weight copy.
    """
    tile = 256 if k % 256 == 0 else 128
    expected = logical_weight.reshape(n, k // tile, tile // 8).transpose(0, 1)
    actual = packed_weight.reshape(k // tile, n, tile // 8)
    return torch.equal(actual, expected)


class LightOpAutoAWQLinearMethod(LinearMethodBase):
    """Wrap a vLLM AutoAWQ method and activate LightOp only when eligible."""

    def __init__(self, delegate: LinearMethodBase, quant_config: object) -> None:
        self.delegate = delegate
        self.quant_config = quant_config
        self._params_dtype: torch.dtype | None = None
        self._k = 0
        self._n = 0
        self._lightop_active = False
        self._lightop_fast_prefill = False
        self._lightop_gemm: Callable[..., torch.Tensor] | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.delegate.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self._params_dtype = params_dtype
        self._k = input_size_per_partition
        self._n = sum(output_partition_sizes)
        if hasattr(layer, _LAYOUT_ATTR):
            delattr(layer, _LAYOUT_ATTR)

    def _eligible(self) -> bool:
        return (
            self._params_dtype == torch.float16
            and getattr(self.quant_config, "weight_bits", None) == 4
            and getattr(self.quant_config, "group_size", None) == 128
            and getattr(self.quant_config, "zero_point", None) is True
            and is_lightop_awq_shape_supported(self._k, self._n)
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        selected_layout = getattr(layer, _LAYOUT_ATTR, None)
        previous_prefill = self._lightop_fast_prefill
        self._lightop_active = False
        self._lightop_fast_prefill = False
        self._lightop_gemm = None
        if selected_layout == _DELEGATE_LAYOUT:
            self.delegate.process_weights_after_loading(layer)
            return
        if selected_layout not in (None, _LIGHTOP_LAYOUT):
            raise RuntimeError(
                f"Unknown LightOp AutoAWQ parameter layout: {selected_layout!r}"
            )

        first_processing = selected_layout is None
        if not self._eligible():
            if not first_processing:
                raise RuntimeError(
                    "LightOp AutoAWQ layer became ineligible during reload"
                )
            self.delegate.process_weights_after_loading(layer)
            setattr(layer, _LAYOUT_ATTR, _DELEGATE_LAYOUT)
            return

        try:
            repack, _gemm = _resolve_lightop_awq_ops()
        except (ImportError, AttributeError, OSError) as error:
            if not first_processing:
                raise RuntimeError(
                    "LightOp AutoAWQ backend became unavailable during reload"
                ) from error
            self.delegate.process_weights_after_loading(layer)
            setattr(layer, _LAYOUT_ATTR, _DELEGATE_LAYOUT)
            return

        try:
            weight_trans, scales_zeros = convert_awq_to_lightop_layout(
                layer.qweight,
                layer.qzeros,
                layer.scales,
                self.quant_config.group_size,
            )
            repacked_weight = repack(weight_trans, self._n, self._k)
            tile = 256 if self._k % 256 == 0 else 128
            if (
                not isinstance(repacked_weight, torch.Tensor)
                or repacked_weight.dtype != torch.int32
                or repacked_weight.device != layer.qweight.device
                or repacked_weight.shape != (self._k // tile, self._n * (tile // 8))
                or not repacked_weight.is_contiguous()
            ):
                raise ValueError(
                    "LightOp AutoAWQ repack returned an incompatible tensor"
                )
            native_gemm = torch.ops.vllm.hcu_lightop_autoawq
            fast_prefill = _matches_prefill_layout(
                weight_trans, repacked_weight, self._k, self._n
            )
            if not first_processing and fast_prefill != previous_prefill:
                raise RuntimeError(
                    "LightOp AutoAWQ prefill layout changed during reload"
                )
            new_weight = Parameter(repacked_weight, requires_grad=False)
            new_scales_zeros = Parameter(scales_zeros, requires_grad=False)
        except (
            TypeError,
            ValueError,
            AssertionError,
            RuntimeError,
            AttributeError,
            OSError,
        ):
            if not first_processing:
                raise
            self.delegate.process_weights_after_loading(layer)
            setattr(layer, _LAYOUT_ATTR, _DELEGATE_LAYOUT)
            return

        layer.qweight = new_weight
        layer.register_parameter("scales_zeros", new_scales_zeros)
        delattr(layer, "qzeros")
        delattr(layer, "scales")
        # Activation registers the dispatcher/FakeTensor boundary before model
        # construction. Never trace the vendor pybind function through Dynamo.
        self._lightop_gemm = native_gemm
        self._lightop_active = True
        self._lightop_fast_prefill = fast_prefill
        setattr(layer, _LAYOUT_ATTR, _LIGHTOP_LAYOUT)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._lightop_active:
            return self.delegate.apply(layer, x, bias)
        if self._lightop_gemm is None:
            raise RuntimeError("LightOp AutoAWQ GEMM was not initialized")

        output_shape = x.shape[:-1] + (self._n,)
        if x.numel() == 0:
            return x.new_empty(output_shape)
        output = self._lightop_gemm(
            x.reshape(-1, x.shape[-1]),
            layer.qweight,
            layer.scales_zeros,
            self._lightop_fast_prefill,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(output_shape)


__all__ = [
    "LightOpAutoAWQLinearMethod",
    "convert_awq_to_lightop_layout",
    "is_lightop_awq_shape_supported",
]
