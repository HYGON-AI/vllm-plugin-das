# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep Qwen4Exp PLE input state on its owning first PP stage."""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from types import ModuleType

from vllm_hcu.qwen4_exp_pp import first_stage_owns_all_ple_layers

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model_state"
PATCH_ID = "worker.core_fix.qwen4_exp.first_stage_ple_model_state"
TARGETS = (f"{TARGET_MODULE}.Qwen4ExpModelState.__init__",)
_MARKER = "_vllm_hcu_qwen4_exp_first_stage_ple_model_state_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_first_stage_ple_model_state_wrapper"
_ORIGINAL = "_vllm_hcu_original_init"
_REQUIRED_NAMES = frozenset({"pipeline_parallel_size", "ple_layer_ids"})
_REQUIRED_CONSTANT = (
    "N-gram PLE embedding currently requires pipeline_parallel_size=1 "
    "because non-first pipeline ranks do not receive the raw input_ids "
    "required by PLE. Please run with PP=1."
)
_V028_INIT_SOURCE_SHA256 = (
    "a068a68c9d6c05beb6d1b5e7139a8ca3c68aa604a0f34bf8d1f00d062193fbd4"
)


def _require_source_fingerprint(function, target: str, expected: str) -> None:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint could "
            f"not be computed: expected sha256={expected}, "
            f"actual=<unavailable>; {type(exc).__name__}: {exc}"
        ) from exc
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint "
            f"mismatch: expected sha256={expected}, actual sha256={actual}"
        )


def _require_audited_init(state_class: type):
    init = vars(state_class).get("__init__")
    if not callable(init):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    require_exact_signature(
        init,
        TARGETS[0],
        positional=(
            "self",
            "vllm_config",
            "model",
            "encoder_cache",
            "device",
        ),
    )
    code = getattr(init, "__code__", None)
    if code is None or not _REQUIRED_NAMES.issubset(code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible source contract"
        )
    if _REQUIRED_CONSTANT not in code.co_consts:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer has the audited "
            "Qwen4Exp PLE pipeline guard"
        )
    _require_source_fingerprint(init, TARGETS[0], _V028_INIT_SOURCE_SHA256)
    return init


def apply_to_module(module: ModuleType) -> bool:
    model_state = load_exact_module(TARGET_MODULE, module)
    state_class = require_class(
        model_state,
        "Qwen4ExpModelState",
        f"{TARGET_MODULE}.Qwen4ExpModelState",
    )
    if getattr(state_class, _MARKER, False):
        current = vars(state_class).get("__init__")
        if not callable(current) or not getattr(current, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    init = _require_audited_init(state_class)

    @functools.wraps(init)
    def hcu_init(self, vllm_config, model, encoder_cache, device) -> None:
        if not first_stage_owns_all_ple_layers(vllm_config):
            return init(self, vllm_config, model, encoder_cache, device)

        from vllm.distributed.parallel_state import get_pp_group

        text_config = vllm_config.model_config.hf_text_config
        if get_pp_group().is_first_rank:
            parallel_config = vllm_config.parallel_config
            pp_size = parallel_config.pipeline_parallel_size
            try:
                parallel_config.pipeline_parallel_size = 1
                return init(self, vllm_config, model, encoder_cache, device)
            finally:
                parallel_config.pipeline_parallel_size = pp_size

        ple_layer_ids = text_config.ple_layer_ids
        try:
            # Later stages contain no PLE modules under the validated layout,
            # so they do not allocate or prepare raw-token PLE inputs.
            text_config.ple_layer_ids = []
            return init(self, vllm_config, model, encoder_cache, device)
        finally:
            text_config.ple_layer_ids = ple_layer_ids

    setattr(hcu_init, _WRAPPER, True)
    setattr(state_class, _ORIGINAL, init)
    setattr(state_class, "__init__", hcu_init)
    setattr(state_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
