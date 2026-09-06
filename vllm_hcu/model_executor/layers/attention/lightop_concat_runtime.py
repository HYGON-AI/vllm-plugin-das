# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""LightOp MLA decode concatenation behind the vLLM HCU runtime contract."""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)


class LightOpMlaConcatUnavailable(RuntimeError):
    """Raised when the selected LightOp package lacks categorized ``ds_cat``."""


def _load_lightop_ds_cat() -> Callable[..., torch.Tensor]:
    try:
        from lightop.tensor import ds_cat
    except (ImportError, AttributeError) as exc:
        raise LightOpMlaConcatUnavailable(
            "lightop.tensor.ds_cat is unavailable"
        ) from exc
    return ds_cat


def lightop_mla_decode_concat_impl(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Execute categorized LightOp ``ds_cat`` mode 0 for MLA decode."""

    output = torch.empty(
        (*left.shape[:-1], left.shape[-1] + right.shape[-1]),
        dtype=left.dtype,
        device=left.device,
    )
    _load_lightop_ds_cat()(left, right, output, 0)
    return output


def lightop_mla_decode_concat_fake(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Describe the allocation contract to ``torch.compile``."""

    return torch.empty(
        (*left.shape[:-1], left.shape[-1] + right.shape[-1]),
        dtype=left.dtype,
        device=left.device,
    )


direct_register_custom_op(
    op_name="lightop_mla_decode_concat",
    op_func=lightop_mla_decode_concat_impl,
    mutates_args=[],
    fake_impl=lightop_mla_decode_concat_fake,
)


def is_lightop_mla_decode_concat_eligible(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    """Match the BW1100 MLA decode layout validated for LightOp mode 0."""

    return (
        left.device.type == "cuda"
        and right.device == left.device
        and left.dtype == torch.bfloat16
        and right.dtype == left.dtype
        and left.dim() == 3
        and right.dim() == 3
        and left.shape[:-1] == right.shape[:-1]
        and left.shape[-1] == 512
        and right.shape[-1] == 64
        and 0 < left.shape[0] < 1024
        and left.shape[1] > 0
        and left.stride(-1) == 1
        and right.stride(-1) == 1
    )


def _call_registered_lightop(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.lightop_mla_decode_concat(left, right)


def concat_mla_decode(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Concatenate MLA query parts, with a fully reversible LightOp route."""

    normalized_dim = dim if dim >= 0 else left.dim() + dim
    if normalized_dim != left.dim() - 1:
        raise ValueError("MLA decode concatenation supports only the last dimension")

    enabled = (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT
        # Preserve the existing public opt-out while the HCU-prefixed leaf is
        # introduced. Both switches default on; the master always wins.
        and henvs.VLLM_USE_OPT_CAT
    )
    if not enabled or not is_lightop_mla_decode_concat_eligible(left, right):
        return torch.cat((left, right), dim=normalized_dim)

    try:
        _load_lightop_ds_cat()
    except LightOpMlaConcatUnavailable:
        logger.warning_once(
            "LightOp MLA decode concat is unavailable; using torch.cat."
        )
        return torch.cat((left, right), dim=normalized_dim)
    logger.warning_once("Using LightOp MLA decode concatenation.")
    return _call_registered_lightop(left, right)


__all__ = [
    "LightOpMlaConcatUnavailable",
    "concat_mla_decode",
    "is_lightop_mla_decode_concat_eligible",
    "lightop_mla_decode_concat_fake",
    "lightop_mla_decode_concat_impl",
]
