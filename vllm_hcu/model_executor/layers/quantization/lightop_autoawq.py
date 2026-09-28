# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Optional public-LightOp AutoAWQ W4A16 linear method."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase


_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
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

    shifts = torch.tensor(
        _AWQ_PACK_ORDER,
        dtype=torch.int32,
        device=qweight.device,
    ) * 4
    logical_weight = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(k, n)
    weight_trans = torch.sum(
        (logical_weight.T.reshape(n, k // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()

    logical_zeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(
        expected_groups, n
    )
    zero_scale_pairs = torch.stack(
        (
            (logical_zeros + 64).to(torch.float16),
            scales.to(torch.float16),
        ),
        dim=-1,
    )
    scales_zeros = (
        zero_scale_pairs.contiguous()
        .view(torch.int32)
        .squeeze(-1)
        .T.contiguous()
    )
    return weight_trans, scales_zeros


def _resolve_lightop_awq_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from lightop.gemm_ops import (
        awq_gemm_marlin_weight_repack,
        gemm_awq_w4a16_marlin,
    )

    return awq_gemm_marlin_weight_repack, gemm_awq_w4a16_marlin


class LightOpAutoAWQLinearMethod(LinearMethodBase):
    """Wrap a vLLM AutoAWQ method and activate LightOp only when eligible."""

    def __init__(self, delegate: LinearMethodBase, quant_config: object) -> None:
        self.delegate = delegate
        self.quant_config = quant_config
        self._params_dtype: torch.dtype | None = None
        self._k = 0
        self._n = 0
        self._lightop_active = False
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

    def _eligible(self) -> bool:
        return (
            self._params_dtype == torch.float16
            and getattr(self.quant_config, "weight_bits", None) == 4
            and getattr(self.quant_config, "group_size", None) == 128
            and getattr(self.quant_config, "zero_point", None) is True
            and is_lightop_awq_shape_supported(self._k, self._n)
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self._eligible():
            self.delegate.process_weights_after_loading(layer)
            return

        try:
            repack, gemm = _resolve_lightop_awq_ops()
        except (ImportError, AttributeError):
            self.delegate.process_weights_after_loading(layer)
            return

        weight_trans, scales_zeros = convert_awq_to_lightop_layout(
            layer.qweight,
            layer.qzeros,
            layer.scales,
            self.quant_config.group_size,
        )
        try:
            repacked_weight = repack(weight_trans, self._n, self._k)
        except (TypeError, ValueError, AssertionError):
            self.delegate.process_weights_after_loading(layer)
            return

        layer.qweight = Parameter(repacked_weight, requires_grad=False)
        layer.register_parameter(
            "scales_zeros", Parameter(scales_zeros, requires_grad=False)
        )
        delattr(layer, "qzeros")
        delattr(layer, "scales")
        self._lightop_gemm = gemm
        self._lightop_active = True

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
        output = self._lightop_gemm(
            x.reshape(-1, x.shape[-1]),
            layer.qweight,
            layer.scales_zeros,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(output_shape)


__all__ = [
    "LightOpAutoAWQLinearMethod",
    "convert_awq_to_lightop_layout",
    "is_lightop_awq_shape_supported",
]
