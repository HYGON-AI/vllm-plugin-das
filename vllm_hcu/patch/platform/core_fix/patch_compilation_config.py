# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU CUDAGraph alignment without extending CompilationConfig."""

from __future__ import annotations

import functools
import inspect
import threading
import weakref
from collections.abc import Mapping
from types import ModuleType

from vllm_hcu.deepseek_v4_runtime import (
    config_field,
    dspark_mooncake_pd_use_low_latency,
)
from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.config.compilation"
PATCH_ID = "platform.core_fix.hcu_config.compilation_cudagraph"
TARGETS = (
    f"{TARGET_MODULE}.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode",
    f"{TARGET_MODULE}.CompilationConfig.set_splitting_ops_for_v1",
)
_MARKER = "_vllm_hcu_compilation_cudagraph_patch_applied"
_HCU_CUDAGRAPH_UNSAFE_SPLITTING_OPS = (
    "vllm::hcu_sparse_attn_indexer",
)
_BOUND_CONFIGS_LOCK = threading.RLock()
_BOUND_CONFIGS: dict[
    int,
    tuple[
        weakref.ReferenceType[object],
        HcuFeatureConfig,
        bool | None,
        int,
    ],
] = {}


def _bound_policy(
    compilation_config: object,
) -> tuple[HcuFeatureConfig, bool | None, int] | None:
    with _BOUND_CONFIGS_LOCK:
        entry = _BOUND_CONFIGS.get(id(compilation_config))
        if entry is None:
            return None
        reference, feature_config, fixed_use_low_latency, parallel_size = entry
        if reference() is compilation_config:
            return feature_config, fixed_use_low_latency, parallel_size
        _BOUND_CONFIGS.pop(id(compilation_config), None)
        return None


def _bound_hcu_config(compilation_config: object) -> HcuFeatureConfig | None:
    policy = _bound_policy(compilation_config)
    return None if policy is None else policy[0]


def _parallel_size(vllm_config: object) -> int:
    parallel_config = config_field(vllm_config, "parallel_config")
    return max(
        int(config_field(parallel_config, name, 1) or 1)
        for name in (
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    )


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
        _BOUND_CONFIGS[key] = (
            reference,
            feature_config,
            dspark_mooncake_pd_use_low_latency(vllm_config),
            _parallel_size(vllm_config),
        )
    return feature_config


def _required_cudagraph_modes(cudagraph_mode_type: object) -> tuple[object, ...]:
    names = (
        "NONE",
        "FULL",
        "PIECEWISE",
        "FULL_AND_PIECEWISE",
        "FULL_DECODE_ONLY",
    )
    modes = tuple(getattr(cudagraph_mode_type, name, None) for name in names)
    if any(mode is None for mode in modes):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CUDAGraphMode is missing "
            + ", ".join(names)
        )
    return modes


def _normalize_deepep_auto_cudagraph_mode(
    current_mode: object,
    *,
    feature_config: HcuFeatureConfig,
    fixed_use_low_latency: bool | None,
    parallel_size: int,
    modes: tuple[object, ...],
) -> tuple[object, str | None]:
    if not feature_config.deepep_auto or parallel_size <= 1:
        return current_mode, None

    none, full, piecewise, full_and_piecewise, full_decode_only = modes
    if fixed_use_low_latency is False and current_mode is not none:
        return none, "fixed_ht"
    if current_mode is full or current_mode is full_and_piecewise:
        return full_decode_only, "decode_only"
    if current_mode is piecewise:
        return none, "piecewise"
    return current_mode, None


def _log_deepep_auto_graph_policy(reason: str | None) -> None:
    if reason is None:
        return
    from vllm.logger import init_logger

    messages = {
        "fixed_ht": (
            "DeepEP auto Mooncake producer uses NONE because fixed HT "
            "dispatch does not support CUDA Graph capture on HCU."
        ),
        "piecewise": (
            "DeepEP auto uses NONE for mixed/prefill execution because HT "
            "PIECEWISE capture is unsupported on HCU."
        ),
        "decode_only": (
            "DeepEP auto uses FULL_DECODE_ONLY so only LL decode execution "
            "is captured on HCU."
        ),
    }
    init_logger(__name__).info_once(messages[reason])


def normalize_bound_deepep_auto_cudagraph_mode(vllm_config: object) -> object:
    """Reapply the bound DeepEP-auto graph policy after platform rewrites."""

    if isinstance(vllm_config, Mapping):
        compilation_config = vllm_config.get("compilation_config")
    else:
        compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        raise PatchCompatibilityError("VllmConfig.compilation_config is missing")
    policy = _bound_policy(compilation_config)
    if policy is None:
        bind_hcu_config(vllm_config)
        policy = _bound_policy(compilation_config)
    if policy is None:
        raise PatchCompatibilityError("HCU CompilationConfig binding is missing")

    from vllm.config.compilation import CUDAGraphMode

    feature_config, fixed_use_low_latency, parallel_size = policy
    effective_mode, reason = _normalize_deepep_auto_cudagraph_mode(
        getattr(compilation_config, "cudagraph_mode", None),
        feature_config=feature_config,
        fixed_use_low_latency=fixed_use_low_latency,
        parallel_size=parallel_size,
        modes=_required_cudagraph_modes(CUDAGraphMode),
    )
    compilation_config.cudagraph_mode = effective_mode
    _log_deepep_auto_graph_policy(reason)
    return effective_mode


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
    original_set_splitting_ops = vars(compilation_config).get(
        "set_splitting_ops_for_v1"
    )
    if not callable(original) or not callable(original_set_splitting_ops):
        raise PatchCompatibilityError(
            "required HCU CompilationConfig compatibility methods are missing"
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

    splitting_signature = inspect.signature(original_set_splitting_ops)
    if tuple(splitting_signature.parameters) != (
        "self",
        "all2all_backend",
        "data_parallel_size",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {splitting_signature}"
        )
    cudagraph_mode_type = getattr(compilation_module, "CUDAGraphMode", None)
    cudagraph_modes = _required_cudagraph_modes(cudagraph_mode_type)

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

    @functools.wraps(original_set_splitting_ops)
    def hcu_set_splitting_ops_for_v1(
        self,
        all2all_backend: str,
        data_parallel_size: int = 1,
    ):
        policy = _bound_policy(self)
        if policy is None:
            policy = HcuFeatureConfig(), None, 1
        feature_config, fixed_use_low_latency, bound_parallel_size = policy
        self.cudagraph_mode, reason = _normalize_deepep_auto_cudagraph_mode(
            getattr(self, "cudagraph_mode", None),
            feature_config=feature_config,
            fixed_use_low_latency=fixed_use_low_latency,
            parallel_size=max(data_parallel_size, bound_parallel_size),
            modes=cudagraph_modes,
        )
        _log_deepep_auto_graph_policy(reason)

        result = original_set_splitting_ops(
            self,
            all2all_backend,
            data_parallel_size,
        )

        # With Inductor graph partitioning enabled, the operator's
        # ``cudagraph_unsafe`` tag is authoritative. HCU currently uses the
        # legacy Dynamo splitting path, so target vLLM's manually maintained
        # splitting list must also contain every HCU-only unsafe operator.
        # GLM DSA reaches this path through ``hcu_sparse_attn_indexer`` during
        # piecewise capture, so the HCU registration is part of its contract.
        if getattr(self, "use_inductor_graph_partition", False):
            return result

        cudagraph_mode = getattr(self, "cudagraph_mode", None)
        has_piecewise = getattr(cudagraph_mode, "has_piecewise_cudagraphs", None)
        if not callable(has_piecewise) or not has_piecewise():
            return result

        splitting_ops = getattr(self, "splitting_ops", None)
        if not isinstance(splitting_ops, list):
            raise PatchCompatibilityError(
                "CompilationConfig.splitting_ops must be finalized as a list "
                "before HCU unsafe operators are registered"
            )
        for op_name in _HCU_CUDAGRAPH_UNSAFE_SPLITTING_OPS:
            if op_name not in splitting_ops:
                splitting_ops.append(op_name)
        return result

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
    setattr(
        compilation_config,
        "_vllm_hcu_original_set_splitting_ops_for_v1",
        original_set_splitting_ops,
    )
    setattr(
        compilation_config,
        "set_splitting_ops_for_v1",
        hcu_set_splitting_ops_for_v1,
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
    "normalize_bound_deepep_auto_cudagraph_mode",
]
