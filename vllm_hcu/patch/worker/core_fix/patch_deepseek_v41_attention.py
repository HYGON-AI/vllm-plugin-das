# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt the DSV4.1 compressor projection to the HCU weight layout."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.compressor_weight_layout"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV4Attention._run_parallel_input_projections",
    f"{TARGET_MODULE}.DeepseekV4Attention.__init__",
)
_CLASS_MARKER = "_vllm_hcu_dsv41_compressor_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_dsv41_compressor_layout_wrapper"


def _compressor_mm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply HCU ``[in, out]`` or upstream ``[out, in]`` weights."""

    if hidden_states.device.type == "cuda":
        from vllm_hcu.model_executor.layers.dsv41_compressor_projection import try_compressor_projection

        result = try_compressor_projection(hidden_states, weight)
        if result is not None:
            return result
    rhs = weight if weight.shape[0] == hidden_states.shape[-1] else weight.T
    return torch.mm(hidden_states, rhs, out_dtype=torch.float32)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "DeepseekV4Attention", f"{TARGET_MODULE}.DeepseekV4Attention")
    original = require_callable(cls, "_run_parallel_input_projections", TARGETS[0])
    require_exact_signature(original, TARGETS[0], positional=("self", "hidden_states"))

    if getattr(cls, _CLASS_MARKER, False):
        for method_name in ("_run_parallel_input_projections", "__init__"):
            if not getattr(vars(cls).get(method_name), _WRAPPER_MARKER, False):
                raise PatchCompatibilityError(
                    f"required HCU patch marker for {PATCH_ID} is stale"
                )
        return False

    original_init = require_callable(cls, "__init__", TARGETS[1])
    require_exact_signature(
        original_init,
        TARGETS[1],
        positional=(
            "self", "vllm_config", "prefix", "topk_indices_buffer",
            "aux_stream_list", "candidate_block_buffer",
        ),
        defaults={
            "topk_indices_buffer": None,
            "aux_stream_list": None,
            "candidate_block_buffer": None,
        },
    )

    execute_in_parallel = require_callable(
        target, "execute_in_parallel", f"{TARGET_MODULE}.execute_in_parallel"
    )
    envs = getattr(target, "envs", None)
    if envs is None:
        raise PatchCompatibilityError(f"required HCU patch target {TARGET_MODULE}.envs is missing")

    @functools.wraps(original_init)
    def hcu_init(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
        candidate_block_buffer=None,
    ):
        if not henvs.VLLM_HCU_ENABLE_DEEPSEEK_V4_MULTI_STREAM:
            aux_stream_list = None
        original_init(
            self, vllm_config, prefix, topk_indices_buffer,
            aux_stream_list, candidate_block_buffer,
        )

    @functools.wraps(original)
    def hcu_run_parallel_input_projections(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            aux_streams = aux_streams[:2]

        aux_fns: list[Callable[[], Any] | None] = [None, None]
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

            aux_fns[1] = indexer_weights_proj

        qr_kv, (kv_score, indexer_weights) = execute_in_parallel(
            lambda: self._fused_wqa_wkv_gemm(hidden_states),
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:3],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )
        return qr_kv, kv_score, indexer_weights

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(cls, "_vllm_hcu_original_attention_init", original_init)
    setattr(cls, "__init__", hcu_init)
    setattr(hcu_run_parallel_input_projections, _WRAPPER_MARKER, True)
    setattr(cls, "_vllm_hcu_original_run_parallel_input_projections", original)
    setattr(cls, "_run_parallel_input_projections", hcu_run_parallel_input_projections)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
