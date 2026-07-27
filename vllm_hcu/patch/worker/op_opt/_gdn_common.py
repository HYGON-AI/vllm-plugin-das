# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared, side-effect-free helpers for vLLM v0.25.1 GDN adapters."""

from __future__ import annotations

from ._common import PatchCompatibilityError


def use_nn_layout() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_USE_NN)


def shape_dim(tensor, index: int) -> int | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[index])
    except (IndexError, TypeError, ValueError):
        return None


def normalize_nn_conv_weight(weight, expected_dim: int | None, target: str):
    if not use_nn_layout():
        return weight
    ndim = getattr(weight, "ndim", None)
    if ndim != 2 or expected_dim is None:
        raise RuntimeError(
            f"HCU GDN NN-layout requires a 2D conv weight for {target}"
        )
    first = shape_dim(weight, 0)
    second = shape_dim(weight, 1)
    if second == expected_dim:
        return weight.transpose(0, 1).contiguous()
    if first == expected_dim:
        return weight.contiguous()
    raise RuntimeError(
        f"HCU GDN NN-layout conv weight for {target} is incompatible: "
        f"weight shape={tuple(weight.shape)!r}, expected dim={expected_dim}"
    )


def require_parameter_names(function, target: str, expected: tuple[str, ...]) -> None:
    import inspect

    try:
        actual = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect required target {target}") from exc
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU target {target} has incompatible parameters {actual!r}"
        )


__all__ = [
    "normalize_nn_conv_weight",
    "require_parameter_names",
    "shape_dim",
    "use_nn_layout",
]
