# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Consume private V4 attention output with suffix-only inverse RoPE."""
from __future__ import annotations
import functools
from types import ModuleType
import torch
from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs
from ._common import (PatchCompatibilityError, load_exact_module, require_callable,
                      require_class, require_exact_signature)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4.inverse_rope_suffix"
TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention._o_proj"
_MARKER = "_vllm_hcu_inverse_rope_suffix_applied"
_WRAPPER_MARKER = "_vllm_hcu_inverse_rope_suffix_wrapper"
logger = init_logger(__name__)


def _eligible(layer, o, positions):
    cache = layer.rotary_emb.cos_sin_cache
    return (henvs.optional_custom_op_enabled(henvs.VLLM_HCU_INVERSE_ROPE_SUFFIX)
        and torch.version.hip is not None and not torch.is_grad_enabled()
        and o.is_cuda and o.dtype == torch.bfloat16 and o.ndim == 3
        and o.shape[0] >= 512 and o.shape[-1] == 512
        and layer.rope_head_dim == 64
        and o.shape[1] == layer.n_local_heads == layer.padded_heads
        and o.is_contiguous() and o.stride(-1) == 1 and o.stride(1) >= 512
        and o.stride(0) >= o.shape[1] * o.stride(1)
        and positions.device == o.device and positions.ndim == 1
        and positions.shape[0] == o.shape[0] and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and cache.device == o.device and cache.ndim == 2
        and cache.dtype == torch.float32 and cache.shape[1] == 64
        and cache.stride(-1) == 1)


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    cls = require_class(rocm, "DeepseekV4ROCMAiterMLAAttention", TARGET_SYMBOL)
    original = require_callable(cls, "_o_proj", TARGET_SYMBOL)
    if getattr(cls, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(f"stale patch for {TARGET_SYMBOL}")
        return False
    require_exact_signature(original, TARGET_SYMBOL, positional=("self", "o", "positions"), defaults={})

    @functools.wraps(original)
    def output_projection(self, o, positions):
        if not _eligible(self, o, positions):
            return original(self, o, positions)
        from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
        from vllm_hcu.utils.deepseek_v4_inverse_rope import inverse_rope_suffix_inplace
        # Only this private consuming projection mutates o. The upstream public
        # inverse-RoPE helper remains out-of-place for all other callers.
        # Resolve weights before mutation, preserving the existing NN adapter.
        grouped_dim = o.shape[1] * o.shape[2] // self.n_local_groups
        weight = ops._get_cached_wo_a_bf16(
            self.wo_a, self.n_local_groups, self.o_lora_rank, grouped_dim)
        inverse_rope_suffix_inplace(o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        z = torch.einsum("tgd,grd->tgr", o.view(o.shape[0], self.n_local_groups, -1), weight)
        zf = z.flatten(1)
        logger.info_once("DeepSeek V4 uses suffix-only inplace inverse RoPE")
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        return self.wo_b(zf)

    setattr(output_projection, _WRAPPER_MARKER, True)
    cls._o_proj = output_projection
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGET_SYMBOL", "apply", "apply_to_module"]
