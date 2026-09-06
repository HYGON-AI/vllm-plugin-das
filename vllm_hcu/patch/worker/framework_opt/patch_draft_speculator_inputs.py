# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid per-step H2D by reusing caller temperature/seeds buffers."""

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

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.draft_input_zero_copy"
TARGETS = (f"{TARGET_MODULE}.DraftModelSpeculator._copy_request_inputs",)
_MARKER = "_vllm_hcu_draft_input_zero_copy_applied"
_WRAPPER = "_vllm_hcu_draft_input_zero_copy_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        spec, "DraftModelSpeculator", f"{TARGET_MODULE}.DraftModelSpeculator"
    )
    wrapped = ((cls, "_copy_request_inputs", TARGETS[0], _WRAPPER),)
    if already_applied(spec, _MARKER, wrapped):
        return False
    original = require_callable(cls, "_copy_request_inputs", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "num_reqs", "idx_mapping", "temperature", "seeds"),
    )

    @functools.wraps(original)
    def hcu_copy_request_inputs(self, num_reqs, idx_mapping, temperature, seeds):
        # Alias caller buffers (often UVA views) instead of H2D copy_.
        # idx_mapping still copied into the fixed CG-padded buffer.
        self.temperature = temperature
        self.seeds = seeds
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        if self.draft_logits is not None:
            self.idx_mapping[num_reqs:].fill_(-1)

    setattr(hcu_copy_request_inputs, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_copy_request_inputs", original)
    setattr(cls, "_copy_request_inputs", hcu_copy_request_inputs)
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
