# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow Qwen4Exp PLE pipeline parallelism when stage zero owns every PLE."""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from types import ModuleType

from vllm_hcu.qwen4_exp_pp import first_stage_owns_all_ple_layers

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.model_executor.models.config"
PATCH_ID = "platform.core_fix.qwen4_exp.first_stage_ple_pp"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpForConditionalGenerationConfig."
    "verify_and_update_config",
)
_MARKER = "_vllm_hcu_qwen4_exp_first_stage_ple_pp_applied"
_ORIGINAL = "_vllm_hcu_original_verify_and_update_config"
_WRAPPER = "_vllm_hcu_qwen4_exp_first_stage_ple_pp_wrapper"
_REQUIRED_NAMES = frozenset({"pipeline_parallel_size", "ple_layer_ids"})
_REQUIRED_CONSTANT = (
    "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
    "because non-first pipeline ranks do not receive the raw input_ids it "
    "needs. Please run with PP=1."
)
_V028_VERIFY_SOURCE_SHA256 = (
    "25d97c784e0cdf27a237cde85380b9bcc70c4826d9e4e4e7dd81f290b8d31a1c"
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


def _require_audited_verifier(config_class: type):
    descriptor = vars(config_class).get("verify_and_update_config")
    if not isinstance(descriptor, staticmethod):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} must be a staticmethod"
        )
    verifier = descriptor.__func__
    try:
        parameters = tuple(inspect.signature(verifier).parameters.values())
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {TARGETS[0]}"
        ) from exc
    if (
        len(parameters) != 1
        or parameters[0].name != "vllm_config"
        or parameters[0].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible signature "
            f"{inspect.signature(verifier)}"
        )
    code = getattr(verifier, "__code__", None)
    if code is None or not _REQUIRED_NAMES.issubset(code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible source contract"
        )
    if _REQUIRED_CONSTANT not in code.co_consts:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer has the audited "
            "Qwen4Exp PLE pipeline guard"
        )
    _require_source_fingerprint(verifier, TARGETS[0], _V028_VERIFY_SOURCE_SHA256)
    return verifier


def apply_to_module(module: ModuleType) -> bool:
    models_config = load_exact_module(TARGET_MODULE, module)
    config_class = getattr(
        models_config, "Qwen4ExpForConditionalGenerationConfig", None
    )
    if not isinstance(config_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            f"{TARGET_MODULE}.Qwen4ExpForConditionalGenerationConfig is missing"
        )
    if getattr(config_class, _MARKER, False):
        descriptor = vars(config_class).get("verify_and_update_config")
        current = descriptor.__func__ if isinstance(descriptor, staticmethod) else None
        if current is None or not getattr(current, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    verifier = _require_audited_verifier(config_class)

    @functools.wraps(verifier)
    def hcu_verify_and_update_config(vllm_config) -> None:
        if not first_stage_owns_all_ple_layers(vllm_config):
            return verifier(vllm_config)

        parallel_config = vllm_config.parallel_config
        pp_size = parallel_config.pipeline_parallel_size
        try:
            # The audited upstream method uses this field only for its blanket
            # PLE guard. Preserve every other Qwen4Exp validation while proving
            # to that guard that no non-first stage owns a PLE layer.
            parallel_config.pipeline_parallel_size = 1
            return verifier(vllm_config)
        finally:
            parallel_config.pipeline_parallel_size = pp_size

    setattr(hcu_verify_and_update_config, _WRAPPER, True)
    setattr(config_class, _ORIGINAL, staticmethod(verifier))
    setattr(
        config_class,
        "verify_and_update_config",
        staticmethod(hcu_verify_and_update_config),
    )
    setattr(config_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
