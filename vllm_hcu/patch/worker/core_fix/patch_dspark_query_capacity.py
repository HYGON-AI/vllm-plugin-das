# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reserve DSpark query capacity independently of the target token budget."""

from __future__ import annotations

import copy
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

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.core_fix.dspark.query_capacity"
_MARKER = "_vllm_hcu_dspark_query_capacity_applied"


def _draft_capacity_config(vllm_config):
    speculative = vllm_config.speculative_config
    hf_config = speculative.draft_model_config.hf_config
    query_width = speculative.num_speculative_tokens + (
        not getattr(hf_config, "sample_from_anchor", True)
    )
    scheduler = vllm_config.scheduler_config
    capacity = max(
        scheduler.max_num_batched_tokens, scheduler.max_num_seqs * query_width
    )
    if capacity == scheduler.max_num_batched_tokens:
        return vllm_config
    # Preserve target scheduling and capture choices; only the drafter and its
    # model/metadata allocations receive the expanded token capacity.
    config = copy.copy(vllm_config)
    config.scheduler_config = copy.copy(scheduler)
    config.scheduler_config.max_num_batched_tokens = capacity
    return config


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator")
    if getattr(cls, _MARKER, False):
        for name in ("__init__", "set_attn"):
            if not getattr(getattr(cls, name), _MARKER, False):
                raise PatchCompatibilityError(f"stale DSpark capacity patch: {name}")
        return False
    original_init = require_callable(cls, "__init__", f"{TARGET_MODULE}.__init__")
    original_set_attn = require_callable(cls, "set_attn", f"{TARGET_MODULE}.set_attn")
    require_exact_signature(
        original_init,
        f"{TARGET_MODULE}.DSparkSpeculator.__init__",
        positional=("self", "vllm_config", "device"),
    )
    require_exact_signature(
        original_set_attn,
        f"{TARGET_MODULE}.DSparkSpeculator.set_attn",
        positional=(
            "self",
            "model_state",
            "kv_cache_config",
            "block_tables",
            "target_input_buffers",
            "target_attn_groups",
        ),
    )

    @functools.wraps(original_init)
    def hcu_init(self, vllm_config, device):
        original_init(self, _draft_capacity_config(vllm_config), device)

    @functools.wraps(original_set_attn)
    def hcu_set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        # Query preparation writes the shared slot mappings. Resize before any
        # graph capture so target and draft retain one stable allocation.
        slots = block_tables.slot_mappings
        if slots.shape[1] < self.max_num_tokens:
            from vllm.v1.attention.backends.utils import PAD_SLOT_ID

            expanded = torch.full(
                (slots.shape[0], self.max_num_tokens),
                PAD_SLOT_ID,
                dtype=slots.dtype,
                device=slots.device,
            )
            expanded[:, : slots.shape[1]].copy_(slots)
            block_tables.slot_mappings = expanded
            block_tables.max_num_batched_tokens = self.max_num_tokens
        return original_set_attn(
            self,
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

    for wrapper in (hcu_init, hcu_set_attn):
        setattr(wrapper, _MARKER, True)
    cls.__init__ = hcu_init
    cls.set_attn = hcu_set_attn
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
