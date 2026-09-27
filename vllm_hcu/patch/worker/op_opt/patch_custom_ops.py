# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route E5M2 MLA cache gather to the HCU-supported LightOp kernel."""

from __future__ import annotations

import functools
from types import ModuleType

import torch
from vllm.logger import init_logger

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm._custom_ops"
PATCH_ID = "worker.op_opt.custom_ops.e5m2_mla_cache_gather"
TARGETS = (f"{TARGET_MODULE}.gather_and_maybe_dequant_cache",)
_MARKER = "_vllm_hcu_e5m2_mla_cache_gather_applied"
_WRAPPER = "_vllm_hcu_e5m2_mla_cache_gather_wrapper"
logger = init_logger(__name__)


def _torch_gather_e5m2_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None,
) -> None:
    """Portable E5M2 MLA gather used when optional custom ops are disabled."""

    if src_cache.dtype == torch.uint8:
        cache = src_cache.view(torch.float8_e5m2)
    elif src_cache.dtype == torch.float8_e5m2:
        cache = src_cache
    else:
        raise TypeError(
            "E5M2 MLA cache must use uint8 or torch.float8_e5m2 storage, "
            f"got {src_cache.dtype}"
        )
    if num_tokens < 0 or num_tokens > token_to_seq.numel():
        raise ValueError(
            f"num_tokens={num_tokens} exceeds token_to_seq capacity "
            f"{token_to_seq.numel()}"
        )
    token_ids = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=token_to_seq.device,
    )
    sequence_ids = token_to_seq[:num_tokens].to(torch.int64)
    sequence_offsets = token_ids - cu_seq_lens[sequence_ids].to(torch.int64)
    if seq_starts is not None:
        sequence_offsets = sequence_offsets + seq_starts[sequence_ids].to(
            torch.int64
        )
    block_size = int(src_cache.shape[1])
    block_columns = torch.div(
        sequence_offsets,
        block_size,
        rounding_mode="floor",
    )
    slot_ids = torch.remainder(sequence_offsets, block_size)
    block_ids = block_table[sequence_ids, block_columns].to(torch.int64)
    gathered = cache[block_ids, slot_ids].to(dst.dtype)
    dst[:num_tokens].copy_(gathered * scale)


def apply_to_module(module: ModuleType) -> bool:
    custom_ops = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (
            custom_ops,
            "gather_and_maybe_dequant_cache",
            TARGETS[0],
            _WRAPPER,
        ),
    )
    if already_applied(custom_ops, _MARKER, wrapped):
        return False

    original = require_callable(
        custom_ops,
        "gather_and_maybe_dequant_cache",
        TARGETS[0],
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "src_cache",
            "dst",
            "block_table",
            "cu_seq_lens",
            "token_to_seq",
            "num_tokens",
            "kv_cache_dtype",
            "scale",
            "seq_starts",
        ),
        defaults={"seq_starts": None},
    )

    @functools.wraps(original)
    def hcu_gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts=None,
    ):
        if kv_cache_dtype != "fp8_e5m2":
            return original(
                src_cache,
                dst,
                block_table,
                cu_seq_lens,
                token_to_seq,
                num_tokens,
                kv_cache_dtype,
                scale,
                seq_starts,
            )
        from vllm_hcu.platforms import envs as henvs

        if not henvs.optional_custom_op_enabled():
            logger.info_once(
                "VLLM_HCU_USE_CUSTOM_OPS=0: using the vllm_hcu PyTorch "
                "E5M2 MLA cache-gather fallback"
            )
            return _torch_gather_e5m2_cache(
                src_cache,
                dst,
                block_table,
                cu_seq_lens,
                token_to_seq,
                num_tokens,
                scale,
                seq_starts,
            )
        try:
            from lightop import gather_and_maybe_dequant_cache as lightop_gather
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "LightOp E5M2 MLA cache gather operator is required but unavailable"
            ) from exc
        return lightop_gather(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            num_tokens,
            kv_cache_dtype,
            scale,
            seq_starts,
        )

    setattr(hcu_gather_and_maybe_dequant_cache, _WRAPPER, True)
    setattr(
        custom_ops,
        "_vllm_hcu_original_gather_and_maybe_dequant_cache",
        original,
    )
    setattr(
        custom_ops,
        "gather_and_maybe_dequant_cache",
        hcu_gather_and_maybe_dequant_cache,
    )
    setattr(custom_ops, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
