# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, single-owner registration for LightOp per-token FP8 quantization."""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch


class HcuLightOpRegistrationError(RuntimeError):
    """The requested LightOp custom operator could not be registered."""


_LOCK = threading.RLock()
_FP8_DTYPE: torch.dtype | None = None
_REGISTERED = False
_REGISTRATION_ERROR: str | None = None


def _lightop_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FP8_DTYPE is None:
        raise HcuLightOpRegistrationError("LightOp FP8 dtype was not initialized")
    try:
        from lightop.quant import per_token_quant_fp8
    except (ImportError, AttributeError) as exc:
        raise HcuLightOpRegistrationError(
            "lightop.quant.per_token_quant_fp8 is required; upgrade LightOp"
        ) from exc

    out = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    try:
        output, output_scale = per_token_quant_fp8(
            x,
            dtype=_FP8_DTYPE,
            out_q=out,
            out_scale=scale,
        )
    except Exception as exc:
        raise HcuLightOpRegistrationError(
            "LightOp per_token_quant_fp8 kernel execution failed"
        ) from exc
    return output, output_scale


def _lightop_per_token_quant_fp8_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FP8_DTYPE is None:
        raise HcuLightOpRegistrationError("LightOp FP8 dtype was not initialized")
    return (
        torch.empty_like(x, dtype=_FP8_DTYPE),
        torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32),
    )


def ensure_registered(
    fp8_dtype: torch.dtype,
    register_custom_op: Callable[..., object],
) -> None:
    """Register exactly once; failures are latched for the process lifetime."""

    global _FP8_DTYPE, _REGISTERED, _REGISTRATION_ERROR
    with _LOCK:
        if _REGISTRATION_ERROR is not None:
            raise HcuLightOpRegistrationError(
                "LightOp per-token FP8 registration previously failed: "
                f"{_REGISTRATION_ERROR}"
            )
        if _REGISTERED:
            if _FP8_DTYPE is not fp8_dtype:
                raise HcuLightOpRegistrationError(
                    f"LightOp per-token FP8 was registered for {_FP8_DTYPE}, "
                    f"not {fp8_dtype}"
                )
            return
        if not callable(register_custom_op):
            raise TypeError("register_custom_op must be callable")
        _FP8_DTYPE = fp8_dtype
        try:
            register_custom_op(
                op_name="lightop_per_token_quant_fp8",
                op_func=_lightop_per_token_quant_fp8,
                mutates_args=[],
                fake_impl=_lightop_per_token_quant_fp8_fake,
            )
        except Exception as exc:
            _REGISTRATION_ERROR = f"{type(exc).__name__}: {exc}"
            raise HcuLightOpRegistrationError(
                "failed to register the HCU-owned "
                "vllm::lightop_per_token_quant_fp8 operator"
            ) from exc
        _REGISTERED = True


def quantize(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    register_custom_op: Callable[..., object],
) -> tuple[torch.Tensor, torch.Tensor]:
    ensure_registered(fp8_dtype, register_custom_op)
    try:
        return torch.ops.vllm.lightop_per_token_quant_fp8(x)
    except Exception as exc:
        raise HcuLightOpRegistrationError(
            "registered HCU LightOp per-token FP8 operator is unavailable"
        ) from exc


def _reset_for_tests() -> None:
    """Reset Python state only; tests must use a fake registrar/operator."""

    global _FP8_DTYPE, _REGISTERED, _REGISTRATION_ERROR
    with _LOCK:
        _FP8_DTYPE = None
        _REGISTERED = False
        _REGISTRATION_ERROR = None


__all__ = [
    "HcuLightOpRegistrationError",
    "ensure_registered",
    "quantize",
]
