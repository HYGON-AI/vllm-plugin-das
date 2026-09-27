# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route E5M2 MLA cache gather to the HCU-supported LightOp kernel."""

from __future__ import annotations

import functools
from types import ModuleType

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
