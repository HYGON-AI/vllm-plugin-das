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

_SUPPORTED_ROUTE_SHAPES = frozenset(
    {
        (256, 8),
        (512, 8),
        (768, 8),
        (128, 16),
        (256, 16),
        (512, 16),
        (768, 16),
        (64, 32),
        (128, 32),
        (256, 32),
        (512, 32),
        (768, 32),
    }
)


class LightOpMlaConcatUnavailable(RuntimeError):
    """Raised when the selected LightOp package lacks categorized ``ds_cat``."""


def _load_lightop_ds_cat() -> Callable[..., torch.Tensor]:
    try:
        from lightop.tensor import ds_cat
    except (ImportError, AttributeError) as exc:
        raise LightOpMlaConcatUnavailable(
            "lightop.tensor.ds_cat is unavailable"
        ) from exc
    if not callable(ds_cat):
        raise LightOpMlaConcatUnavailable(
            "lightop.tensor.ds_cat is unavailable"
        )
    return ds_cat


def lightop_mla_decode_concat_impl(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Execute categorized LightOp ``ds_cat`` mode 0 for MLA decode."""

    try:
        ds_cat = _load_lightop_ds_cat()
    except LightOpMlaConcatUnavailable:
        logger.warning_once(
            "LightOp MLA decode concat is unavailable; using torch.cat."
        )
        return torch.cat((left, right), dim=-1)
    output = torch.empty(
        (*left.shape[:-1], left.shape[-1] + right.shape[-1]),
        dtype=left.dtype,
        device=left.device,
    )
    ds_cat(left, right, output, 0)
    logger.warning_once("Using LightOp MLA decode concatenation.")
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


def is_lightop_mla_decode_concat_shape_supported(tokens: int, heads: int) -> bool:
    """Return whether production-route performance passed for this shape."""

    return (int(tokens), int(heads)) in _SUPPORTED_ROUTE_SHAPES


def is_lightop_mla_decode_concat_eligible(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    """Match the BW1100 MLA decode layout validated for LightOp mode 0."""

    if (
        left.device.type == "cuda"
        and right.device == left.device
        and left.dtype == torch.bfloat16
        and right.dtype == left.dtype
        and left.dim() == 3
        and right.dim() == 3
        and left.shape[:-1] == right.shape[:-1]
        and left.shape[-1] == 512
        and right.shape[-1] == 64
    ):
        tokens = int(left.shape[0])
        heads = int(left.shape[1])
        return bool(
            is_lightop_mla_decode_concat_shape_supported(tokens, heads)
            and tuple(left.stride(index) for index in range(3))
            == (512, 512 * tokens, 1)
            and tuple(right.stride(index) for index in range(3))
            == (1536 * (heads // 8), 192, 1)
        )
    return False


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

    return _call_registered_lightop(left, right)


__all__ = [
    "LightOpMlaConcatUnavailable",
    "concat_mla_decode",
    "is_lightop_mla_decode_concat_eligible",
    "is_lightop_mla_decode_concat_shape_supported",
    "lightop_mla_decode_concat_fake",
    "lightop_mla_decode_concat_impl",
]
