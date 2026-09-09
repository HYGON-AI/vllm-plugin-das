# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Strict LightOp fast path for Qwen gated RMSNorm."""

from __future__ import annotations

import functools
import importlib

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.utils.torch_utils import direct_register_custom_op

import vllm_hcu.platforms.envs as henvs


logger = init_logger(__name__)

# Verified Qwen3.5/Qwen3.6 decode and chunked-prefill geometries. Unmeasured
# shapes retain vLLM's canonical Triton path.
_VERIFIED_QWEN_ROWS = frozenset(
    {
        1,
        2,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        144,
        192,
        256,
        384,
        512,
        768,
        1024,
    }
)
_VERIFIED_QWEN_WIDTHS = frozenset({128, 256})


@functools.lru_cache(maxsize=1)
def _lightop_layer_norm_fwd_1pass_opt():
    try:
        module = importlib.import_module("lightop.norm")
    except ImportError:
        return None
    function = getattr(module, "layer_norm_fwd_1pass_opt", None)
    return function if callable(function) else None


def _rows_per_block(rows: int, compute_units: int) -> int:
    value = max(1, (rows + 2 * compute_units - 1) // (2 * compute_units))
    return min(1 << (value - 1).bit_length(), 4)


def _vllm_qwen_rmsnorm_gated_fallback(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

    return rmsnorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=eps,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )


def _hcu_lightop_qwen_rmsnorm_gated_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    layer_norm_fwd_1pass_opt = _lightop_layer_norm_fwd_1pass_opt()
    if layer_norm_fwd_1pass_opt is None:
        return _vllm_qwen_rmsnorm_gated_fallback(x, z, weight, eps)

    rows, width = x.shape
    output = torch.empty_like(x)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    compute_units = torch.cuda.get_device_properties(x.device).multi_processor_count
    with torch.cuda.device(x.device):
        layer_norm_fwd_1pass_opt(
            x,
            output,
            weight,
            None,
            z,
            None,
            rstd,
            x.stride(0),
            output.stride(0),
            z.stride(0),
            rows,
            width,
            eps,
            width,
            _rows_per_block(rows, compute_units),
            False,
            True,
            True,
            True,
            "silu",
        )
    logger.warning_once("Using LightOp Qwen gated RMSNorm.")
    return output


def _hcu_lightop_qwen_rmsnorm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del z, weight, eps
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="hcu_lightop_qwen_rmsnorm_gated",
    op_func=_hcu_lightop_qwen_rmsnorm_gated_impl,
    fake_impl=_hcu_lightop_qwen_rmsnorm_gated_fake,
)


def _is_qwen_gated_rmsnorm_eligible(
    layer: RMSNormGated,
    x: torch.Tensor,
    z: torch.Tensor | None,
) -> bool:
    if not (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED
    ):
        return False
    if (
        x.ndim != 2
        or x.shape[0] not in _VERIFIED_QWEN_ROWS
        or x.shape[1] not in _VERIFIED_QWEN_WIDTHS
    ):
        return False
    if z is None or z.shape != x.shape:
        return False
    weight = layer.weight
    if (
        x.device.type != "cuda"
        or weight.ndim != 1
        or weight.shape[0] != x.shape[1]
    ):
        return False
    if (
        x.dtype is not torch.bfloat16
        or z.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
    ):
        return False
    if not (x.is_contiguous() and z.is_contiguous() and weight.is_contiguous()):
        return False
    if x.device != z.device or x.device != weight.device:
        return False
    effective_group_size = (
        x.shape[-1] if layer.group_size is None else layer.group_size
    )
    return bool(
        layer.bias is None
        and effective_group_size == x.shape[-1]
        and layer.norm_before_gate is True
        and layer.activation == "silu"
    )


class HcuRMSNormGated(RMSNormGated):
    def forward_hip(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _is_qwen_gated_rmsnorm_eligible(self, x, z):
            assert z is not None
            return torch.ops.vllm.hcu_lightop_qwen_rmsnorm_gated(
                x,
                z,
                self.weight,
                self.eps,
            )
        return self.forward_cuda(x, z)


__all__ = [
    "HcuRMSNormGated",
    "_hcu_lightop_qwen_rmsnorm_gated_impl",
    "_is_qwen_gated_rmsnorm_eligible",
    "_lightop_layer_norm_fwd_1pass_opt",
]
