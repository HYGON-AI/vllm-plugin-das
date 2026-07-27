# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Retire ModelRunnerOutput IPC mutation in favour of DraftTokenIds."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.v1.outputs"
PATCH_ID = "platform.framework_opt.outputs_draft_token_ids"
TARGETS = (
    f"{TARGET_MODULE}.ModelRunnerOutput",
    f"{TARGET_MODULE}.DraftTokenIds",
    f"{TARGET_MODULE}.EMPTY_MODEL_RUNNER_OUTPUT",
)
_MARKER = "_vllm_hcu_draft_token_ids_contract_validated"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    model_output = require_class(target, "ModelRunnerOutput", TARGETS[0])
    draft_ids = require_class(target, "DraftTokenIds", TARGETS[1])
    if not is_dataclass(model_output) or not is_dataclass(draft_ids):
        raise PatchCompatibilityError("vLLM output contracts must be dataclasses")
    model_fields = {field.name for field in fields(model_output)}
    if "spec_token_ids" in model_fields:
        raise PatchCompatibilityError(
            "ModelRunnerOutput was source-patched with spec_token_ids; clean vLLM is required"
        )
    draft_fields = tuple(field.name for field in fields(draft_ids))
    if draft_fields != ("req_ids", "draft_token_ids"):
        raise PatchCompatibilityError(
            f"DraftTokenIds has incompatible fields: {draft_fields!r}"
        )
    empty = getattr(target, "EMPTY_MODEL_RUNNER_OUTPUT", None)
    if not isinstance(empty, model_output):
        raise PatchCompatibilityError("EMPTY_MODEL_RUNNER_OUTPUT has wrong type")
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [ "PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
