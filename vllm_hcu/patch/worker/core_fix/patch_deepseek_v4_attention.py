# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Make DeepSeek V4 compressor GEMMs accept the HCU NN weight layout."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from ._common import load_exact_module, require_callable, require_class

TARGET_MODULE = "vllm.models.deepseek_v4.attention"
PATCH_ID = "worker.core_fix.deepseek_v4.attention_compressor_weight_layout"
TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4Attention.attn_gemm_parallel_execute"
_CLASS_MARKER = "_vllm_hcu_compressor_weight_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_compressor_weight_layout_wrapper"


def _compressor_mm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply either HCU NN ``[in, out]`` or upstream ``[out, in]`` weights."""
    rhs = weight if weight.shape[0] == hidden_states.shape[-1] else weight.T
    return torch.mm(hidden_states, rhs, out_dtype=torch.float32)


def apply_to_module(module: ModuleType) -> bool:
    attention = load_exact_module(TARGET_MODULE, module)
    cls = require_class(attention, "DeepseekV4Attention", TARGET_SYMBOL)
    original = require_callable(cls, "attn_gemm_parallel_execute", TARGET_SYMBOL)
    if getattr(cls, _CLASS_MARKER, False):
        return False

    execute_in_parallel = require_callable(
        attention, "execute_in_parallel", f"{TARGET_MODULE}.execute_in_parallel"
    )
    envs = attention.envs

    @functools.wraps(original)
    def hcu_attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        aux_fns: list[Callable[[], Any] | None] = [None, None, None]
        if self.compressor is not None:
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return _compressor_mm(
                    hidden_states, compressor.fused_wkv_wgate.weight
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return _compressor_mm(
                    hidden_states, indexer.compressor.fused_wkv_wgate.weight
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=(
                hidden_states.shape[0]
                <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
            ),
        )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    setattr(hcu_attn_gemm_parallel_execute, _WRAPPER_MARKER, True)
    setattr(cls, "attn_gemm_parallel_execute", hcu_attn_gemm_parallel_execute)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
