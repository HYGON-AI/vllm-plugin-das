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
    cudagraph_none = getattr(cudagraph_mode_type, "NONE", None)
    if cudagraph_none is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CUDAGraphMode.NONE is missing"
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

    @functools.wraps(original_set_splitting_ops)
    def hcu_set_splitting_ops_for_v1(
        self,
        all2all_backend: str,
        data_parallel_size: int = 1,
    ):
        result = original_set_splitting_ops(
            self,
            all2all_backend,
            data_parallel_size,
        )
        feature_config = _bound_hcu_config(self) or HcuFeatureConfig()
        if feature_config.deepep_auto and data_parallel_size > 1:
            # deepep_auto includes the same HT dispatch that upstream marks
            # CUDA-Graph incompatible for DP. Keep the standard CLI contract
            # and apply upstream's automatic graph fallback internally.
            if getattr(self, "cudagraph_mode", None) != cudagraph_none:
                logger = getattr(compilation_module, "logger", None)
                if logger is not None:
                    logger.info(
                        "DeepEP auto: disabling CUDA Graphs because the "
                        "high-throughput dispatch phase is not graph compatible."
                    )
                self.cudagraph_mode = cudagraph_none
            return result

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
]
