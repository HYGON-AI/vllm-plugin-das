# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Custom-SP cudagraph alignment without extending CompilationConfig."""

from __future__ import annotations

import functools
import inspect
import threading
import weakref
from collections.abc import Mapping
from types import ModuleType

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.config.compilation"
PATCH_ID = "platform.core_fix.hcu_config.compilation_custom_sp"
TARGETS = (
    f"{TARGET_MODULE}.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode",
)
_MARKER = "_vllm_hcu_custom_sp_cudagraph_patch_applied"
_BOUND_CONFIGS_LOCK = threading.RLock()
_BOUND_CONFIGS: dict[
    int,
    tuple[weakref.ReferenceType[object], HcuFeatureConfig],
] = {}


def _bound_hcu_config(compilation_config: object) -> HcuFeatureConfig | None:
    with _BOUND_CONFIGS_LOCK:
        entry = _BOUND_CONFIGS.get(id(compilation_config))
        if entry is None:
            return None
        reference, feature_config = entry
        if reference() is compilation_config:
            return feature_config
        _BOUND_CONFIGS.pop(id(compilation_config), None)
        return None


def bind_hcu_config(vllm_config: object) -> HcuFeatureConfig:
    """Bind the immutable sidecar to CompilationConfig for its narrow callback.

    The binding lives only in this module's weak-reference table; no private
    attribute or dataclass/Pydantic field is added to the upstream config.
    ``VllmConfig.additional_config`` remains the authoritative serialized copy.
    A spawned process must bind its deserialized VllmConfig before using the
    compilation callback.
    """

    feature_config = get_hcu_config(vllm_config)
    if isinstance(vllm_config, Mapping):
        compilation_config = vllm_config.get("compilation_config")
    else:
        compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        raise PatchCompatibilityError("VllmConfig.compilation_config is missing")
    key = id(compilation_config)

    def remove_binding(reference: weakref.ReferenceType[object]) -> None:
        with _BOUND_CONFIGS_LOCK:
            current = _BOUND_CONFIGS.get(key)
            if current is not None and current[0] is reference:
                _BOUND_CONFIGS.pop(key, None)

    try:
        reference = weakref.ref(compilation_config, remove_binding)
    except TypeError as exc:
        raise PatchCompatibilityError(
            "CompilationConfig must support weak references for HCU binding"
        ) from exc
    with _BOUND_CONFIGS_LOCK:
        _BOUND_CONFIGS[key] = (reference, feature_config)
    return feature_config


def apply_to_module(module: ModuleType) -> bool:
    compilation_module = load_exact_module(TARGET_MODULE, module)
    compilation_config = getattr(compilation_module, "CompilationConfig", None)
    if not isinstance(compilation_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CompilationConfig is missing"
        )
    if getattr(compilation_config, _MARKER, False):
        return False

    original = vars(compilation_config).get(
        "adjust_cudagraph_sizes_for_spec_decode"
    )
    if not callable(original):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    signature = inspect.signature(original)
    if tuple(signature.parameters) != (
        "self",
        "uniform_decode_query_len",
        "tensor_parallel_size",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    @functools.wraps(original)
    def hcu_adjust_cudagraph_sizes(
        self,
        uniform_decode_query_len: int,
        tensor_parallel_size: int,
    ):
        # The process-local binding avoids adding a second serialized copy to
        # CompilationConfig.__dict__.  VllmConfig.additional_config remains
        # authoritative and dispatchers rebind after spawn/unpickle.
        feature_config = _bound_hcu_config(self) or HcuFeatureConfig()
        pass_config = getattr(self, "pass_config", None)
        if pass_config is None or not hasattr(pass_config, "enable_sp"):
            raise PatchCompatibilityError(
                "CompilationConfig.pass_config.enable_sp is missing"
            )

        # The feature-off path calls upstream directly and is byte-for-byte
        # behavior equivalent.  For custom-SP, reuse upstream's complete
        # validation/rounding implementation through its existing enable_sp
        # branch, restoring the official config immediately afterwards.
        if (
            not feature_config.enable_custom_sp
            or tensor_parallel_size <= 1
            or pass_config.enable_sp
        ):
            return original(self, uniform_decode_query_len, tensor_parallel_size)

        pass_config.enable_sp = True
        try:
            return original(self, uniform_decode_query_len, tensor_parallel_size)
        finally:
            pass_config.enable_sp = False

    setattr(
        compilation_config,
        "_vllm_hcu_original_adjust_cudagraph_sizes_for_spec_decode",
        original,
    )
    setattr(
        compilation_config,
        "adjust_cudagraph_sizes_for_spec_decode",
        hcu_adjust_cudagraph_sizes,
    )
    setattr(compilation_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    compilation_module = load_exact_module(TARGET_MODULE, module)
    compilation_config = getattr(compilation_module, "CompilationConfig", None)
    if not isinstance(compilation_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CompilationConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=compilation_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(compilation_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "bind_hcu_config",
]
