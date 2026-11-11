# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the existing HCU inverse-RoPE op in the native V4 projection."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4.inverse_rope_suffix"
TARGET_SYMBOL = f"{TARGET_MODULE}.rocm_inv_rope_einsum"
_MARKER = "_vllm_hcu_inverse_rope_suffix_applied"
logger = init_logger(__name__)


@functools.cache
def _register_inverse_rope_fake() -> bool:
    """Register the FakeTensor implementation for the mutating HCU op."""
    try:
        import vllm_hcu.hcu_ops  # noqa: F401

        getattr(torch.ops.hcu_ops, "deepseek_v4_inv_rope")
    except (ImportError, AttributeError):
        return False

    def deepseek_v4_inv_rope_fake(rope, position_ids, cos_sin_cache):
        return None

    try:
        torch.library.register_fake("hcu_ops::deepseek_v4_inv_rope")(
            deepseek_v4_inv_rope_fake
        )
    except RuntimeError as exc:
        if "already has" not in str(exc) and "already registered" not in str(exc):
            raise
    return True


def _eligible(rotary_emb, o, positions, rope_head_dim, n_local_groups):
    cache = rotary_emb.cos_sin_cache
    return (
        henvs.optional_custom_op_enabled(henvs.VLLM_HCU_INVERSE_ROPE_SUFFIX)
        and torch.version.hip is not None
        and not torch.is_grad_enabled()
        and o.is_cuda
        and o.dtype == torch.bfloat16
        and o.ndim == 3
        and o.shape[0] >= 512
        and o.shape[-1] == 512
        and rope_head_dim == 64
        and n_local_groups > 0
        and (o.shape[1] * o.shape[2]) % n_local_groups == 0
        and o.is_contiguous()
        and positions.device == o.device
        and positions.ndim == 1
        and positions.shape[0] == o.shape[0]
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and cache.device == o.device
        and cache.ndim == 2
        and cache.dtype == torch.float32
        and cache.shape[1] == 64
        and cache.stride(-1) == 1
    )


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    _register_inverse_rope_fake()
    original = require_callable(rocm, "rocm_inv_rope_einsum", TARGET_SYMBOL)
    if getattr(rocm, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError(f"stale patch for {TARGET_SYMBOL}")
        return False
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "rotary_emb",
            "o",
            "positions",
            "rope_head_dim",
            "n_local_groups",
            "o_lora_rank",
            "wo_a",
        ),
    )

    @functools.wraps(original)
    def inverse_rope_einsum(
        rotary_emb,
        o,
        positions,
        rope_head_dim,
        n_local_groups,
        o_lora_rank,
        wo_a,
    ):
        if not _eligible(
            rotary_emb, o, positions, rope_head_dim, n_local_groups
        ):
            return original(
                rotary_emb,
                o,
                positions,
                rope_head_dim,
                n_local_groups,
                o_lora_rank,
                wo_a,
            )

        from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

        hidden_dim = o.shape[1] * o.shape[2] // n_local_groups
        weight = ops._get_cached_wo_a_bf16(
            wo_a, n_local_groups, o_lora_rank, hidden_dim
        )
        torch.ops.hcu_ops.deepseek_v4_inv_rope(
            o[..., -rope_head_dim:], positions, rotary_emb.cos_sin_cache
        )
        return torch.einsum(
            "tgd,grd->tgr",
            o.view(o.shape[0], n_local_groups, hidden_dim),
            weight,
        )

    setattr(inverse_rope_einsum, _MARKER, True)
    rocm.rocm_inv_rope_einsum = inverse_rope_einsum
    setattr(rocm, _MARKER, True)
    logger.info_once("DeepSeek V4 uses the HCU suffix inverse-RoPE op")
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGET_SYMBOL", "apply", "apply_to_module"]
