# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Backport the PCP-aware EPLB validation from upstream vLLM."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from pydantic.dataclasses import rebuild_dataclass

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.config.parallel"
PATCH_ID = "platform.core_fix.parallel_config.pcp_eplb"
TARGETS = (f"{TARGET_MODULE}.ParallelConfig._validate_parallel_config",)
_MARKER = "_vllm_hcu_pcp_eplb_validation_applied"
_VALIDATOR_NAME = "_validate_parallel_config"


def apply_to_module(module: ModuleType) -> bool:
    """Allow EPLB when PCP supplies the expert-parallel world size.

    vLLM 0.25.1's validator only counts TP x DP, although its world-size and
    EP-group construction already include PCP.  Reuse the complete pinned
    validator with a temporary TP sentinel for the one obsolete gate.  The
    instance is restored before construction returns, so its topology and
    serialized values remain unchanged.

    Pydantic dataclasses compile validators into a schema.  Updating only the
    Python method therefore has no effect; update the matching decorator entry
    and rebuild the schema transactionally as well.
    """

    parallel_module = load_exact_module(TARGET_MODULE, module)
    parallel_config = getattr(parallel_module, "ParallelConfig", None)
    if not isinstance(parallel_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.ParallelConfig is missing"
        )
    if getattr(parallel_config, _MARKER, False):
        return False

    original = vars(parallel_config).get(_VALIDATOR_NAME)
    if not callable(original):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    decorator_infos = getattr(
        parallel_config, "__pydantic_decorators__", None
    )
    validators = getattr(decorator_infos, "model_validators", None)
    if not isinstance(validators, dict):
        raise PatchCompatibilityError(
            "ParallelConfig Pydantic model-validator registry is missing"
        )
    validator = validators.get(_VALIDATOR_NAME)
    if validator is None or getattr(validator, "func", None) is not original:
        raise PatchCompatibilityError(
            "ParallelConfig Pydantic validator does not match the audited "
            "vLLM 0.25.1 contract"
        )
    validator_info = getattr(validator, "info", None)
    if getattr(validator_info, "mode", None) != "after":
        raise PatchCompatibilityError(
            "ParallelConfig validation must remain an after-model validator"
        )

    @functools.wraps(original)
    def hcu_validate_parallel_config(self):
        enable_pcp_eplb = (
            bool(getattr(self, "enable_eplb", False))
            and int(getattr(self, "tensor_parallel_size", 1))
            * int(getattr(self, "data_parallel_size", 1))
            <= 1
            and int(getattr(self, "prefill_context_parallel_size", 1)) > 1
        )
        if not enable_pcp_eplb:
            return original(self)

        tensor_parallel_size = self.tensor_parallel_size
        self.tensor_parallel_size = 2
        try:
            validated = original(self)
        finally:
            self.tensor_parallel_size = tensor_parallel_size

        # The sentinel must not accidentally relax the separate v0.25.1 DCP
        # constraint.  HCU PCP currently supports DCP=1, but retaining this
        # check here also keeps direct ParallelConfig construction compatible.
        if tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                f"tp_size={tensor_parallel_size} must be divisible by"
                f"dcp_size={self.decode_context_parallel_size}."
            )
        return validated

    setattr(
        parallel_config,
        "_vllm_hcu_original_validate_parallel_config",
        original,
    )
    setattr(parallel_config, _VALIDATOR_NAME, hcu_validate_parallel_config)
    validator.func = hcu_validate_parallel_config
    try:
        rebuilt = rebuild_dataclass(parallel_config, force=True)
        if rebuilt is not True:
            raise PatchCompatibilityError(
                "ParallelConfig Pydantic schema could not be rebuilt"
            )
    except Exception as exc:
        setattr(parallel_config, _VALIDATOR_NAME, original)
        validator.func = original
        rebuild_dataclass(parallel_config, force=True)
        if isinstance(exc, PatchCompatibilityError):
            raise
        raise PatchCompatibilityError(
            "ParallelConfig Pydantic schema rebuild failed"
        ) from exc

    setattr(parallel_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    parallel_module = load_exact_module(TARGET_MODULE, module)
    parallel_config = getattr(parallel_module, "ParallelConfig", None)
    if not isinstance(parallel_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.ParallelConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=parallel_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(parallel_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
