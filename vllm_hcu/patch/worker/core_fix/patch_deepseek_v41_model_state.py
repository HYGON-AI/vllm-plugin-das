# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gather Engram history from PCP fragment starts rather than request starts."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.nvidia.model_state"
PATCH_ID = "worker.core_fix.deepseek_v41.pcp_lookback"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV41ModelState.__init__",
    f"{TARGET_MODULE}.DeepseekV41ModelState.prepare_inputs",
)
_MARKER = "_vllm_hcu_dsv41_pcp_lookback_applied"
_WRAPPER = "_vllm_hcu_dsv41_pcp_lookback_wrapper"


def _fill_fragment_lookback(window, input_batch, req_states) -> None:
    count = input_batch.num_reqs
    if count > window.shape[0]:
        raise RuntimeError(
            f"PCP Engram lookback capacity exceeded: {count} > {window.shape[0]}"
        )
    starts = torch.as_tensor(
        input_batch.num_computed_tokens_np[:count], device=window.device
    )
    positions = (
        starts[:, None] - 1 - torch.arange(window.shape[1], device=window.device)
    )
    indices = input_batch.idx_mapping[:count].long()
    history = req_states.all_token_ids.gpu[
        indices[:, None], positions.clamp_min(0).long()
    ]
    window[:count].copy_(torch.where(positions >= 0, history, -1))
    window[count:].fill_(-1)


def apply_to_module(module: ModuleType) -> bool:
    model_state = load_exact_module(TARGET_MODULE, module)
    state_class = require_class(
        model_state, "DeepseekV41ModelState", f"{TARGET_MODULE}.DeepseekV41ModelState"
    )
    if getattr(state_class, _MARKER, False):
        if not all(
            getattr(vars(state_class).get(name), _WRAPPER, False)
            for name in ("__init__", "prepare_inputs")
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {PATCH_ID} is stale"
            )
        return False
    original_init = require_callable(state_class, "__init__", TARGETS[0])
    original_prepare = require_callable(state_class, "prepare_inputs", TARGETS[1])
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "vllm_config", "model", "encoder_cache", "device"),
    )
    require_exact_signature(
        original_prepare, TARGETS[1], positional=("self", "input_batch", "req_states")
    )
    default_class = require_class(
        model_state, "DefaultModelState", f"{TARGET_MODULE}.DefaultModelState"
    )
    default_prepare = require_callable(
        default_class,
        "prepare_inputs",
        f"{TARGET_MODULE}.DefaultModelState.prepare_inputs",
    )

    @functools.wraps(original_init)
    def hcu_init(self, vllm_config, model, encoder_cache, device):
        original_init(self, vllm_config, model, encoder_cache, device)
        window = self.lookback_token_ids
        if (
            vllm_config.parallel_config.prefill_context_parallel_size > 1
            and window is not None
        ):
            # Match HcuPCPManager's two segments plus one virtual row per request.
            capacity = 2 * ((3 * self.max_num_reqs + 1) // 2)
            self.lookback_token_ids = window.new_full((capacity, window.shape[1]), -1)

    @functools.wraps(original_prepare)
    def hcu_prepare(self, input_batch, req_states):
        if self.vllm_config.parallel_config.prefill_context_parallel_size <= 1:
            return original_prepare(self, input_batch, req_states)
        model_inputs = default_prepare(self, input_batch, req_states)
        window = self.lookback_token_ids
        if window is not None:
            _fill_fragment_lookback(window, input_batch, req_states)
            model_inputs["lookback_token_ids"] = window
        return model_inputs

    for wrapper in (hcu_init, hcu_prepare):
        setattr(wrapper, _WRAPPER, True)
    state_class._vllm_hcu_original_lookback_init = original_init
    state_class._vllm_hcu_original_lookback_prepare = original_prepare
    state_class.__init__ = hcu_init
    state_class.prepare_inputs = hcu_prepare
    setattr(state_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
