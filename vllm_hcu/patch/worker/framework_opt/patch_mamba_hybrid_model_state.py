# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Align resumed Mamba state indices with the hybrid manager block size."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.model_states.mamba_hybrid"
PATCH_ID = "worker.framework_opt.mamba.hybrid_resume_state_index"
TARGETS = (
    f"{TARGET_MODULE}.MambaHybridModelState.add_request",
    f"{TARGET_MODULE}.MambaHybridModelState.preprocess_state",
)
_MARKER = "_vllm_hcu_mamba_hybrid_state_index_applied"
_ADD_WRAPPER = "_vllm_hcu_mamba_hybrid_add_request_wrapper"
_PREPROCESS_WRAPPER = "_vllm_hcu_mamba_hybrid_preprocess_wrapper"
_PENDING_SEEDS = "_vllm_hcu_pending_mamba_state_seeds"


def _seed_state_index(
    self, req_index: int, num_computed_tokens: int, block_size: int
) -> None:
    self._mamba_state_idx_gpu[req_index].fill_(
        (num_computed_tokens - 1) // block_size
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    model_state = require_class(
        target,
        "MambaHybridModelState",
        f"{TARGET_MODULE}.MambaHybridModelState",
    )
    if already_applied(
        target,
        _MARKER,
        (
            (model_state, "add_request", TARGETS[0], _ADD_WRAPPER),
            (
                model_state,
                "preprocess_state",
                TARGETS[1],
                _PREPROCESS_WRAPPER,
            ),
        ),
    ):
        return False

    original_add_request = require_callable(model_state, "add_request", TARGETS[0])
    require_exact_signature(
        original_add_request,
        TARGETS[0],
        positional=("self", "req_index", "new_req_data"),
    )
    original_preprocess_state = require_callable(
        model_state, "preprocess_state", TARGETS[1]
    )
    require_exact_signature(
        original_preprocess_state,
        TARGETS[1],
        positional=(
            "self",
            "input_batch",
            "block_tables",
            "kv_cache_config",
            "num_computed_tokens",
        ),
    )

    @functools.wraps(original_add_request)
    def add_request(self, req_index, new_req_data):
        result = original_add_request(self, req_index, new_req_data)
        if not self._align_mode:
            return result

        num_computed_tokens = new_req_data.num_computed_tokens
        mamba_spec = self._mamba_spec
        if mamba_spec is None:
            pending = getattr(self, _PENDING_SEEDS, None)
            if pending is None:
                pending = {}
                setattr(self, _PENDING_SEEDS, pending)
            pending[req_index] = num_computed_tokens
        elif self.cache_config.block_size != mamba_spec.block_size:
            _seed_state_index(
                self,
                req_index,
                num_computed_tokens,
                mamba_spec.block_size,
            )
        return result

    @functools.wraps(original_preprocess_state)
    def preprocess_state(
        self,
        input_batch,
        block_tables,
        kv_cache_config,
        num_computed_tokens,
    ):
        pending = getattr(self, _PENDING_SEEDS, None)
        if self._align_mode and pending:
            _, mamba_spec = self._get_mamba_group_info(kv_cache_config)
            if self.cache_config.block_size != mamba_spec.block_size:
                for req_index, seed_tokens in pending.items():
                    _seed_state_index(
                        self,
                        req_index,
                        seed_tokens,
                        mamba_spec.block_size,
                    )
            pending.clear()
        return original_preprocess_state(
            self,
            input_batch,
            block_tables,
            kv_cache_config,
            num_computed_tokens,
        )

    setattr(add_request, _ADD_WRAPPER, True)
    setattr(preprocess_state, _PREPROCESS_WRAPPER, True)
    setattr(model_state, "_vllm_hcu_original_add_request", original_add_request)
    setattr(
        model_state,
        "_vllm_hcu_original_preprocess_state",
        original_preprocess_state,
    )
    setattr(model_state, "add_request", add_request)
    setattr(model_state, "preprocess_state", preprocess_state)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
