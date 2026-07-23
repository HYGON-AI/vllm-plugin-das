# SPDX-License-Identifier: Apache-2.0
"""Explicit worker-phase runtime patch dispatcher.

Importing this package is side-effect free.  :func:`prepare_worker_patches`
only arms exact module names; it never imports a target vLLM module or an HCU
runner/custom-op implementation.  :func:`apply_worker_patches` is called when
the deserialised worker config is available and records the real feature state
before ``HcuGPUWorker`` imports model/operator modules.

The inventory is deliberately written out in dependency order.  Directory
scanning would make import order depend on filenames and could let the
official MoE custom-op module load before its HCU replacement.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Literal

from vllm_hcu.compatibility import ensure_vllm_compatible
from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config
from vllm_hcu.patch.import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
    OptionalPatchUnavailable,
)
from vllm_hcu.patch.runtime_state import (
    LatchedPatchError,
    PatchStatus,
    set_process_role,
)


_DISPATCH_LOCK = threading.RLock()
_ADAPTER_ROOT = "vllm_hcu.patch.worker"

FeatureKey = Literal[
    "always",
    "custom_sp",
    "deepep",
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_auto",
    "dpsk_deep_gemm",
    "forward_context",
    "lightly_cp",
    "multi_layers_mtp",
    "proposer",
    "pynccl_all_to_all",
]


@dataclass(frozen=True, slots=True)
class _CallbackSpec:
    adapter: str
    feature: FeatureKey = "always"
    patch_id: str | None = None
    targets_from: int = 0
    optional_capability: bool = False


@dataclass(frozen=True, slots=True)
class _ReplacementSpec:
    adapter: str
    patch_id: str
    target_module: str
    replacement_module: str
    targets: tuple[str, ...]
    feature: FeatureKey = "always"
    validate_with_adapter: bool = True


def _adapter(group: str, name: str) -> str:
    return f"{_ADAPTER_ROOT}.{group}.{name}"


# The two HCU-owned MoE modules must be armed in this exact order.  The runner
# imports SharedExperts and both modules participate in Torch custom-op/model
# construction, so a post-import swap is unsafe.  All import-facing metadata
# is duplicated here intentionally: platform registration must not import an
# adapter before the coordinator's atomic registration fence is active.  A
# test below compares this frozen inventory with the adapters to prevent drift.
_SHARED_EXPERTS_MODULE = (
    "vllm.model_executor.layers.fused_moe.runner.shared_experts"
)
_MOE_RUNNER_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
_MOE_REPLACEMENTS: tuple[_ReplacementSpec, ...] = (
    _ReplacementSpec(
        adapter=_adapter("op_opt.moe", "patch_shared_experts"),
        patch_id="worker.op_opt.moe.runner.shared_experts",
        target_module=_SHARED_EXPERTS_MODULE,
        replacement_module=(
            "vllm_hcu.model_executor.layers.fused_moe.shared_experts"
        ),
        targets=(
            _SHARED_EXPERTS_MODULE,
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._layer_supports_x_and_scale_quanted",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._run_layer",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._disable_shared_experts_overlap",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._determine_shared_experts_order",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._should_run_shared_in_aux_stream",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts.maybe_sync_shared_experts_stream",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._launch_in_aux_stream",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts._run_in_aux_stream",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts.output",
            f"{_SHARED_EXPERTS_MODULE}.SharedExperts.forward",
        ),
    ),
    _ReplacementSpec(
        adapter=_adapter("op_opt.moe", "patch_moe_runner"),
        patch_id="worker.op_opt.moe.runner",
        target_module=_MOE_RUNNER_MODULE,
        replacement_module=(
            "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
        ),
        targets=(
            _MOE_RUNNER_MODULE,
            f"{_MOE_RUNNER_MODULE}._moe_forward",
            f"{_MOE_RUNNER_MODULE}._moe_forward_fake",
            f"{_MOE_RUNNER_MODULE}._moe_forward_shared",
            f"{_MOE_RUNNER_MODULE}._moe_forward_shared_fake",
            f"{_MOE_RUNNER_MODULE}.MoERunner._maybe_apply_shared_experts",
            f"{_MOE_RUNNER_MODULE}.MoERunner._quant_method_supports_quanted_inputs",
            f"{_MOE_RUNNER_MODULE}.MoERunner._apply_quant_method",
            f"{_MOE_RUNNER_MODULE}.MoERunner._maybe_sync_shared_experts_stream",
            f"{_MOE_RUNNER_MODULE}.MoERunner.forward",
            f"{_MOE_RUNNER_MODULE}.MoERunner._forward_impl",
        ),
    ),
)


# ``vllm._aiter_ops`` invokes register_ops_once() at module completion.  It is
# therefore a cold whole-module exchange, never a post-import callback.
_OP_REPLACEMENTS: tuple[_ReplacementSpec, ...] = (
    _ReplacementSpec(
        adapter=_adapter("op_opt", "patch_aiter_ops"),
        patch_id="worker.op_opt.aiter_ops.hcu_runtime",
        target_module="vllm._aiter_ops",
        replacement_module=(
            "vllm_hcu.model_executor.layers.fused_moe.aiter_ops"
        ),
        targets=(
            "vllm._aiter_ops",
            "vllm._aiter_ops.is_aiter_found_and_supported",
            "vllm._aiter_ops._get_aiter_w16a16_moe_solution_id",
            "vllm._aiter_ops._rocm_aiter_fused_moe_impl",
            "vllm._aiter_ops._rocm_aiter_topk_softmax_impl",
            "vllm._aiter_ops.rocm_aiter_ops.get_aiter_activation_type",
            "vllm._aiter_ops.rocm_aiter_ops."
            "is_shuffled_per_token_w8a8_gemm_tuned",
            "vllm._aiter_ops.rocm_aiter_ops.is_per_token_w8a8_gemm_tuned",
            "vllm._aiter_ops.rocm_aiter_ops.is_fp8bmm_enabled",
            "vllm._aiter_ops._rocm_aiter_rmsnorm_fused_dynamic_quant_impl",
            "vllm._aiter_ops."
            "_rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl",
        ),
    ),
)


_CUSTOM_ALLREDUCE_MODULE = (
    "vllm.distributed.device_communicators.custom_all_reduce"
)
_CUSTOM_ALLREDUCE_REPLACEMENT = _ReplacementSpec(
    adapter=_adapter("framework_opt", "patch_cuda_communicator"),
    patch_id="worker.framework_opt.communicator.hcu_custom_allreduce_exchange",
    target_module=_CUSTOM_ALLREDUCE_MODULE,
    replacement_module=(
        "vllm_hcu.distributed.device_communicators.custom_all_reduce"
    ),
    targets=(
        _CUSTOM_ALLREDUCE_MODULE,
        "vllm_hcu.distributed.device_communicators.custom_all_reduce",
    ),
    # The CudaCommunicator callback validates the public integration contract;
    # this replacement itself is a direct lazy module alias.
    validate_with_adapter=False,
)

_COLD_REPLACEMENTS: tuple[_ReplacementSpec, ...] = (
    *_MOE_REPLACEMENTS,
    *_OP_REPLACEMENTS,
    _CUSTOM_ALLREDUCE_REPLACEMENT,
)


# MoE config/oracle utilities are registered before any module which may
# construct a prepare/finalize object, router, expert, runner, or FusedMoE.
_MOE_FOUNDATION_CALLBACKS: tuple[_CallbackSpec, ...] = (
    _CallbackSpec(_adapter("op_opt.moe", "patch_config")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_utils")),
    _CallbackSpec(
        _adapter("op_opt.moe", "patch_fp8_oracle"),
        feature="dpsk_deep_gemm",
    ),
)


_CORE_CALLBACKS: tuple[_CallbackSpec, ...] = (
    _CallbackSpec(_adapter("core_fix", "patch_deepseek_v32_config")),
    _CallbackSpec(_adapter("core_fix", "patch_gpt_oss_mlp_block")),
    _CallbackSpec(_adapter("core_fix", "patch_qwen3_5_mamba_state_dtype")),
    _CallbackSpec(_adapter("core_fix", "patch_qwen3_vl")),
    _CallbackSpec(_adapter("core_fix", "patch_qwen3_vl_moe")),
)


# Attention's public package callback consumes the class installed by the
# ``attention.attention`` callback, hence the non-alphabetical first entries.
_OP_CALLBACKS: tuple[_CallbackSpec, ...] = (
    _CallbackSpec(_adapter("op_opt", "patch_attention_layer")),
    _CallbackSpec(_adapter("op_opt", "patch_attention_exports")),
    _CallbackSpec(_adapter("op_opt", "patch_attention_backend")),
    _CallbackSpec(_adapter("op_opt", "patch_attention_backend_utils")),
    _CallbackSpec(_adapter("op_opt", "patch_triton_unified_attention")),
    _CallbackSpec(_adapter("op_opt", "patch_mla_attention")),
    _CallbackSpec(_adapter("op_opt", "patch_mla_layer"), feature="lightly_cp"),
    _CallbackSpec(_adapter("op_opt", "patch_mla_indexer")),
    _CallbackSpec(_adapter("op_opt", "patch_flashmla_sparse")),
    _CallbackSpec(_adapter("op_opt", "patch_fla_chunk_delta_h")),
    _CallbackSpec(_adapter("op_opt", "patch_fla_chunk_o")),
    _CallbackSpec(_adapter("op_opt", "patch_mamba_mixer2")),
    # All GDN deltas bind Qwen's module-local symbols/class only.  The
    # canonical causal-conv module and the shared GDN base remain vLLM-owned
    # so Kimi, Olmo, MambaMixer, MambaMixer2, and ShortConv are not patched.
    _CallbackSpec(_adapter("op_opt", "patch_gdn_causal_conv1d")),
    _CallbackSpec(_adapter("op_opt", "patch_gdn_base")),
    _CallbackSpec(_adapter("op_opt", "patch_gdn_linear_attention")),
    _CallbackSpec(_adapter("op_opt", "patch_activation")),
    _CallbackSpec(_adapter("op_opt", "patch_layers_utils")),
    _CallbackSpec(_adapter("op_opt", "patch_deep_gemm")),
    _CallbackSpec(_adapter("op_opt", "patch_scaled_mm_linear_kernel")),
    _CallbackSpec(_adapter("op_opt", "patch_input_quant_fp8")),
    _CallbackSpec(_adapter("op_opt", "patch_w8a8_utils")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors_scheme")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors_w8a8_int8")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors_w8a8_fp8")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors_moe_w8a8_fp8")),
    _CallbackSpec(_adapter("op_opt", "patch_compressed_tensors_moe_wna16")),
)


# Experts and routers are armed before the layer that constructs them.
# DeepEP implementations are armed before all2all_utils selects one of them.
_MOE_RUNTIME_CALLBACKS: tuple[_CallbackSpec, ...] = (
    _CallbackSpec(_adapter("op_opt.moe", "patch_fused_moe")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_moe_align_block_size")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_rocm_aiter_moe")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_triton_moe")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_base_router")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_fused_topk_bias_router")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_router_factory")),
    _CallbackSpec(
        _adapter("op_opt.moe", "patch_deepep_ht"),
        feature="deepep_high_throughput",
    ),
    _CallbackSpec(
        _adapter("op_opt.moe", "patch_deepep_ll"),
        feature="deepep_low_latency",
    ),
    _CallbackSpec(
        _adapter("op_opt.moe", "patch_all2all_utils"),
        feature="deepep_low_latency",
    ),
    _CallbackSpec(_adapter("op_opt.moe", "patch_fused_moe_modular_method")),
    _CallbackSpec(_adapter("op_opt.moe", "patch_layer")),
)


# The custom-allreduce exchange has its own replacement registration.  Its
# adapter also validates CudaCommunicator, which needs a distinct patch id
# because one coordinator id cannot describe both a replacement and callback.
_CUDA_VALIDATION_ID = (
    "worker.framework_opt.communicator.hcu_custom_allreduce_exchange."
    "cuda_communicator_validation"
)


# Required dependency order from the worker framework migration:
# dp -> forward context -> base communicator -> PyNccl wrapper -> PyNccl ->
# proposer/eagle/ubatch -> DeepEP all2all.  The PyNccl pair is optional unless
# an explicit ``all2all_backend='pynccl'`` config requests it.
_FRAMEWORK_CALLBACKS: tuple[_CallbackSpec, ...] = (
    _CallbackSpec(
        _adapter("framework_opt", "patch_gpu_worker_shutdown"),
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_dp_utils"),
        feature="deepep_low_latency",
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_forward_context"),
        feature="forward_context",
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_base_device_communicator"),
        feature="custom_sp",
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_pynccl_wrapper"),
        feature="pynccl_all_to_all",
        optional_capability=True,
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_pynccl"),
        feature="pynccl_all_to_all",
        optional_capability=True,
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_llm_base_proposer"),
        feature="proposer",
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_eagle_utils"),
        feature="multi_layers_mtp",
    ),
    _CallbackSpec(_adapter("framework_opt", "patch_gpu_ubatch_wrapper")),
    _CallbackSpec(_adapter("framework_opt", "patch_ubatch_utils")),
    _CallbackSpec(
        _adapter("framework_opt", "patch_all2all"), feature="deepep"
    ),
    _CallbackSpec(
        _adapter("framework_opt", "patch_cuda_communicator"),
        patch_id=_CUDA_VALIDATION_ID,
        targets_from=1,
    ),
)


_ALL_CALLBACK_GROUPS: tuple[tuple[_CallbackSpec, ...], ...] = (
    _MOE_FOUNDATION_CALLBACKS,
    _CORE_CALLBACKS,
    _OP_CALLBACKS,
    _MOE_RUNTIME_CALLBACKS,
    _FRAMEWORK_CALLBACKS,
)


# Only feature-gated atomic-chain nodes belong to terminal validation.  Most
# ``always`` callbacks are model-specific compatibility paths; a dense model
# must not fail merely because a Qwen-VL/GPT-OSS/Mamba module was never
# imported.  These entries, in contrast, are required whenever their recorded
# feature state is true.
_REQUIRED_TERMINAL_IDS = frozenset(
    {
        "worker.op_opt.moe.oracle.fp8_dpsk",
        "worker.op_opt.moe.prepare_finalize.deepep_ht",
        "worker.op_opt.moe.prepare_finalize.deepep_ll",
        "worker.op_opt.moe.all2all_utils",
        "worker.op_opt.mla.lightly_cp_wrapper",
        "worker.framework_opt.dp.deepep_low_latency",
        "worker.framework_opt.forward_context.hcu_runtime_fields",
        "worker.framework_opt.communicator.base_custom_sp",
        "worker.framework_opt.communicator.pynccl_wrapper_all_to_all",
        "worker.framework_opt.communicator.pynccl_all_to_all",
        "worker.framework_opt.spec_decode.hcu_proposer",
        "worker.framework_opt.spec_decode.eagle_topk_buffer",
        "worker.framework_opt.communicator.deep_ep_runtime",
    }
)


def _load_adapter(module_name: str) -> ModuleType:
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "apply_to_module", None)):
        raise TypeError(f"worker patch adapter {module_name!r} has no apply_to_module")
    return module


def _require_bool_result(adapter: ModuleType, result: object, phase: str) -> bool:
    if type(result) is not bool:
        raise TypeError(
            f"{adapter.__name__}.apply_to_module returned "
            f"{type(result).__name__} during {phase}; expected bool"
        )
    return result


def _apply_and_verify(adapter: ModuleType, module: ModuleType) -> None:
    """Apply an adapter and prove that its marker-backed path is idempotent."""

    first = _require_bool_result(
        adapter, adapter.apply_to_module(module), "initial application"
    )
    second = _require_bool_result(
        adapter, adapter.apply_to_module(module), "idempotency validation"
    )
    if second:
        raise RuntimeError(
            f"{adapter.__name__}.apply_to_module reapplied on its marker-backed "
            "validation call"
        )
    # ``first is False`` is valid only for a target whose adapter has already
    # validated its own marker/identity.  Every required adapter performs that
    # check before returning False; the second call catches unstable no-ops.
    del first


def _registration_feature(
    coordinator: ExactImportCoordinator,
    patch_id: str,
    default: bool,
) -> bool:
    # Repeated general-plugin callbacks must not overwrite feature=True after
    # the worker config has arrived.
    for registration in coordinator.registrations():
        if registration.patch_id == patch_id:
            return registration.feature_enabled
    return default


@lru_cache(maxsize=None)
def _callback_for(
    adapter_name: str,
    optional_capability: bool,
    coordinator: ExactImportCoordinator,
    patch_id: str,
) -> Callable[[ModuleType], None]:
    """Return one stable callback object for idempotent re-registration."""

    def callback(module: ModuleType) -> None:
        adapter = _load_adapter(adapter_name)
        if not optional_capability:
            _apply_and_verify(adapter, module)
            return

        # The feature flag may change after prepare() when a spawned worker
        # receives its VllmConfig.  Read it at target-import time instead of
        # capturing the initial feature-off state.
        required = False
        for registration in coordinator.registrations():
            if registration.patch_id == patch_id:
                required = registration.feature_enabled
                break
        first = _require_bool_result(
            adapter,
            adapter.apply_to_module(module, required=required),
            "optional capability probe",
        )
        applied = bool(getattr(module, "_vllm_hcu_pynccl_all_to_all_applied", False))
        if not first and not applied:
            reason = getattr(
                module,
                "_vllm_hcu_pynccl_all_to_all_probe",
                "PyNccl all-to-all capability is unavailable",
            )
            # required=True is expected to raise in the adapter.  Keep this
            # guard fail-closed if a future adapter accidentally returns False.
            if required:
                raise RuntimeError(
                    "PyNccl all-to-all was explicitly requested but is "
                    f"unavailable: {reason}"
                )
            raise OptionalPatchUnavailable(str(reason))
        second = _require_bool_result(
            adapter,
            adapter.apply_to_module(module, required=required),
            "optional capability idempotency validation",
        )
        if second:
            raise RuntimeError(
                f"{adapter.__name__}.apply_to_module repeated optional "
                "capability registration"
            )

    callback.__name__ = f"apply_{patch_id.replace('.', '_')}"
    callback.__qualname__ = callback.__name__
    return callback


@lru_cache(maxsize=None)
def _replacement_factory(adapter_name: str) -> Callable[[], ModuleType]:
    """Import and validate an HCU replacement only on canonical first use."""

    def factory() -> ModuleType:
        adapter = _load_adapter(adapter_name)
        replacement_name = getattr(adapter, "REPLACEMENT_MODULE", None)
        if not isinstance(replacement_name, str) or not replacement_name:
            raise TypeError(
                f"worker replacement adapter {adapter_name!r} has no "
                "REPLACEMENT_MODULE"
            )
        replacement = importlib.import_module(replacement_name)
        _apply_and_verify(adapter, replacement)
        return replacement

    factory.__name__ = f"load_{adapter_name.rsplit('.', 1)[-1]}"
    factory.__qualname__ = factory.__name__
    return factory


def _register_worker_cold_replacements(
    coordinator: ExactImportCoordinator,
) -> list[ImportRegistration]:
    """Register the frozen cold inventory; caller owns any batch fence."""

    registrations: list[ImportRegistration] = []
    for spec in _COLD_REPLACEMENTS:
        replacement = (
            _replacement_factory(spec.adapter)
            if spec.validate_with_adapter
            else spec.replacement_module
        )
        registrations.append(
            coordinator.register_replacement(
                spec.patch_id,
                spec.target_module,
                replacement,
                targets=spec.targets,
                feature_enabled=_registration_feature(
                    coordinator, spec.patch_id, spec.feature == "always"
                ),
                late_policy="fail",
            )
        )
    return registrations


def _register_callbacks(
    coordinator: ExactImportCoordinator,
) -> list[ImportRegistration]:
    registrations: list[ImportRegistration] = []
    for group in _ALL_CALLBACK_GROUPS:
        for spec in group:
            adapter = _load_adapter(spec.adapter)
            patch_id = spec.patch_id or adapter.PATCH_ID
            adapter_targets = getattr(adapter, "TARGETS", ())
            targets = tuple(adapter_targets[spec.targets_from :])
            if not targets:
                # The retired DeepSeek verifier adapter predates TARGETS and
                # intentionally reports its one audited upstream symbol.
                target_symbol = getattr(adapter, "TARGET_SYMBOL", None)
                if not isinstance(target_symbol, str):
                    targets = (adapter.TARGET_MODULE,)
                else:
                    targets = (target_symbol,)
            registrations.append(
                coordinator.register_callback(
                    patch_id,
                    adapter.TARGET_MODULE,
                    _callback_for(
                        spec.adapter,
                        spec.optional_capability,
                        coordinator,
                        patch_id,
                    ),
                    targets=targets,
                    feature_enabled=_registration_feature(
                        coordinator, patch_id, spec.feature == "always"
                    ),
                )
            )
    return registrations


def _known_patch_ids() -> set[str]:
    values = {_CUDA_VALIDATION_ID}
    values.update(spec.patch_id for spec in _COLD_REPLACEMENTS)
    for group in _ALL_CALLBACK_GROUPS:
        for spec in group:
            values.add(spec.patch_id or _load_adapter(spec.adapter).PATCH_ID)
    return values


def _failure_reason(
    coordinator: ExactImportCoordinator, patch_id: str
) -> str | None:
    registry = getattr(coordinator, "_registry", None)
    record = registry.get(patch_id) if registry is not None else None
    return getattr(record, "failure_reason", None)


def _raise_latched_or_required_failures(
    coordinator: ExactImportCoordinator,
) -> None:
    known = _known_patch_ids()
    for registration in coordinator.registrations():
        if registration.patch_id not in known:
            continue
        reason = _failure_reason(coordinator, registration.patch_id)
        if registration.status == PatchStatus.FAILED.value:
            raise LatchedPatchError(
                f"required worker patch {registration.patch_id!r} failed: "
                f"{reason or 'unknown failure'}"
            )
        if (
            registration.feature_enabled
            and registration.status == PatchStatus.SKIPPED.value
        ):
            raise RuntimeError(
                f"worker feature requires patch {registration.patch_id!r}, but "
                f"the capability was skipped: {reason or 'unavailable'}"
            )


def _prepare_worker_cold_replacements(
    coordinator: ExactImportCoordinator | None = None,
) -> tuple[ImportRegistration, ...]:
    """Arm import-time/custom-op replacements without loading implementations.

    Platform discovery calls the underlying static registration helper before
    any platform module exchange can import a canonical dependency such as
    ``vllm._aiter_ops`` or the MoE shared-experts module.  No adapter or HCU
    implementation is imported while the four exact aliases are armed; their
    validation factories remain lazy.  Worker callbacks are deliberately
    configured later by :func:`prepare_worker_patches`.

    The operation is process-local, thread-safe, and idempotent.  It does not
    change the process role or import a target/replacement business module.
    """

    coordinator = IMPORT_COORDINATOR if coordinator is None else coordinator
    with _DISPATCH_LOCK:
        with coordinator.registration_batch():
            coordinator.install()
            return tuple(_register_worker_cold_replacements(coordinator))


def prepare_worker_patches(
    coordinator: ExactImportCoordinator | None = None,
) -> tuple[ImportRegistration, ...]:
    """Arm all worker replacements and callbacks without importing targets.

    General plugins may call this before a worker config exists.  The method
    is process-local, thread-safe, and idempotent.  It intentionally does not
    change the process role: only :func:`apply_worker_patches` knows it is
    running in a real Worker.
    """

    coordinator = IMPORT_COORDINATOR if coordinator is None else coordinator
    with _DISPATCH_LOCK:
        registrations = list(_prepare_worker_cold_replacements(coordinator))
        registrations.extend(_register_callbacks(coordinator))
        _raise_latched_or_required_failures(coordinator)
        return tuple(registrations)


def _parallel_backend(vllm_config: object | None) -> str | None:
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None and isinstance(vllm_config, dict):
        parallel = vllm_config.get("parallel_config")
    if isinstance(parallel, dict):
        backend = parallel.get("all2all_backend")
    else:
        backend = getattr(parallel, "all2all_backend", None)
    return backend if isinstance(backend, str) else None


def _feature_states(
    vllm_config: object | None,
    config: HcuFeatureConfig,
) -> dict[FeatureKey, bool]:
    backend = _parallel_backend(vllm_config)
    deepep_auto = config.deepep_auto
    deepep = backend in {"deepep_high_throughput", "deepep_low_latency"}
    # ``pynccl`` is not silently inferred from custom-SP.  Custom-SP has a
    # portable torch.distributed path; only an explicit backend selection may
    # turn the optional RCCL symbol into a required startup capability.
    pynccl = backend == "pynccl"
    return {
        "always": True,
        "custom_sp": config.enable_custom_sp,
        "deepep": deepep or deepep_auto,
        "deepep_high_throughput": (
            backend == "deepep_high_throughput" or deepep_auto
        ),
        "deepep_low_latency": (
            backend == "deepep_low_latency" or deepep_auto
        ),
        "deepep_auto": deepep_auto,
        "dpsk_deep_gemm": (
            config.moe_backend == "dpsk_deep_gemm" or deepep_auto
        ),
        "forward_context": bool(
            config.enable_lightly_cp
            or backend == "deepep_low_latency"
            or deepep_auto
        ),
        "lightly_cp": config.enable_lightly_cp,
        "multi_layers_mtp": config.enable_multi_layers_mtp,
        "proposer": bool(
            config.enable_lightly_cp or config.enable_multi_layers_mtp
        ),
        "pynccl_all_to_all": pynccl,
    }


def _patch_features() -> dict[str, FeatureKey]:
    features: dict[str, FeatureKey] = {}
    for spec in _COLD_REPLACEMENTS:
        features[spec.patch_id] = spec.feature
    for group in _ALL_CALLBACK_GROUPS:
        for spec in group:
            adapter = _load_adapter(spec.adapter)
            features[spec.patch_id or adapter.PATCH_ID] = spec.feature
    return features


def _bind_deserialized_hcu_config(vllm_config: object) -> HcuFeatureConfig:
    """Rebuild the process-local CompilationConfig binding after IPC."""

    # Keep the platform adapter import at the actual Worker/config boundary;
    # importing this dispatcher must remain side-effect free and must not pull
    # platform adapter packages into general-plugin discovery.
    from vllm_hcu.patch.platform.core_fix.patch_compilation_config import (
        bind_hcu_config,
    )

    return bind_hcu_config(vllm_config)


def apply_worker_patches(vllm_config: object | None = None) -> None:
    """Arm worker patches, bind sidecar feature state, and fail closed.

    ``vllm_config`` may be an object, its pickled dictionary form, or ``None``
    for feature-off tests.  Reading it through :func:`get_hcu_config` performs
    the same strict sidecar validation in spawned workers as in the main
    process.  Target modules remain lazy; an armed required callback will fail
    its importing worker if the audited target vLLM symbol is incompatible.
    """

    ensure_vllm_compatible()

    # Keep this API self-contained for direct Worker construction.  Platform
    # dispatch also acquires the worker cold-replacement lock, so call it
    # before entering the worker critical section to preserve one global lock
    # order under concurrent plugin discovery.
    from vllm_hcu.patch.platform import apply_platform_patches

    apply_platform_patches()
    with _DISPATCH_LOCK:
        # Set Worker *after* platform dispatch so its best-effort role
        # detection cannot overwrite the authoritative role.
        set_process_role("Worker")
        # This is both normalization and the spawn/unpickle boundary check.
        config = get_hcu_config(vllm_config)
        if HcuFeatureConfig.from_mapping(config.to_dict()) != config:
            raise RuntimeError("HCU feature sidecar failed its worker round-trip check")
        if vllm_config is not None:
            rebound = _bind_deserialized_hcu_config(vllm_config)
            if rebound != config:
                raise RuntimeError(
                    "HCU CompilationConfig binding disagrees with the "
                    "deserialized worker sidecar"
                )

        prepare_worker_patches()
        states = _feature_states(vllm_config, config)
        for patch_id, feature in _patch_features().items():
            IMPORT_COORDINATOR.set_feature_enabled(patch_id, states[feature])
        _raise_latched_or_required_failures(IMPORT_COORDINATOR)


def validate_worker_patches(
    require_applied: bool = True,
    *,
    coordinator: ExactImportCoordinator | None = None,
) -> None:
    """Validate enabled feature chains at a caller-defined terminal point.

    Worker construction calls :func:`apply_worker_patches` before model
    imports, where ``armed`` is expected.  After the model/communicator graph
    has loaded, the caller can invoke this function with ``require_applied``
    (the default) to reject an enabled chain whose required callback never ran.
    Feature-off callbacks may remain armed indefinitely.  Model-specific
    always-on compatibility callbacks are intentionally outside this terminal
    set because their modules are not expected in every model family.
    """

    if not isinstance(require_applied, bool):
        raise TypeError("require_applied must be bool")
    coordinator = IMPORT_COORDINATOR if coordinator is None else coordinator
    _raise_latched_or_required_failures(coordinator)
    if not require_applied:
        return

    pending: list[str] = []
    for registration in coordinator.registrations():
        if (
            registration.patch_id in _REQUIRED_TERMINAL_IDS
            and registration.feature_enabled
            and registration.status != PatchStatus.APPLIED.value
        ):
            pending.append(
                f"{registration.patch_id} ({registration.status})"
            )
    if pending:
        raise RuntimeError(
            "enabled HCU worker feature chain did not reach its required "
            "terminal patches: " + ", ".join(sorted(pending))
        )


def worker_callback_names() -> tuple[tuple[str, str], ...]:
    """Return the frozen callback inventory for diagnostics and tests."""

    values: list[tuple[str, str]] = []
    for group in _ALL_CALLBACK_GROUPS:
        for spec in group:
            adapter = _load_adapter(spec.adapter)
            values.append((spec.patch_id or adapter.PATCH_ID, adapter.TARGET_MODULE))
    return tuple(values)


def worker_module_exchange_names() -> tuple[tuple[str, str, str], ...]:
    """Return worker replacement ids in their safety-critical order."""

    return tuple(
        (spec.patch_id, spec.target_module, spec.replacement_module)
        for spec in _COLD_REPLACEMENTS
    )


__all__ = [
    "apply_worker_patches",
    "prepare_worker_patches",
    "validate_worker_patches",
    "worker_callback_names",
    "worker_module_exchange_names",
]
