# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep per-step MTP request-buffer copies asynchronous."""

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
PATCH_ID = "worker.framework_opt.spec_decode.draft_input_async_copy"
TARGETS = (f"{TARGET_MODULE}.DraftModelSpeculator._copy_request_inputs",)
_MARKER = "_vllm_hcu_draft_input_async_copy_applied"
_WRAPPER = "_vllm_hcu_draft_input_async_copy_wrapper"


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
        # Preserve the graph-owned destination addresses. Rebinding these
        # buffers to caller tensors makes captured speculator graphs read stale
        # storage. Copies on the current stream retain ordering without forcing
        # synchronous H2D transfers when the source memory supports async copy.
        self.temperature.copy_(temperature, non_blocking=True)
        self.seeds.copy_(seeds, non_blocking=True)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        # Keep current upstream semantics: every graph-padded request must map
        # to -1 so stale slots cannot update draft logits.
        self.idx_mapping[num_reqs:].fill_(-1)

    setattr(hcu_copy_request_inputs, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_copy_request_inputs", original)
    setattr(cls, "_copy_request_inputs", hcu_copy_request_inputs)
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
