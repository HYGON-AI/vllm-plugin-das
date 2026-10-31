"""Adapter extension lifecycle for migrated DAS optimizations."""

from __future__ import annotations

from importlib import import_module as _import_module
import inspect
import json
import os
import threading

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionSpec,
)

from .environment import register_compile_factors
from .version import TARGET_VLLM_RELEASE


_FLA_CAPABILITY = "boltops-fla-decode"
_CAPABILITIES = frozenset(
    {"lightop-rms-norm", "aiter-int8-moe", _FLA_CAPABILITY}
)
_IMPLEMENTED_EXTENSION_API_VERSION = 1
_AITER_SIGNATURES = {
    "get_aiter_moe_config": (
        "M",
        "E",
        "N1",
        "N2",
        "K",
        "top_k",
        "block_size",
        "dtype",
        "quant_type",
        "activation",
        "gated",
        "spec_sol_type",
        "use_shuffle",
    ),
    "aiter_moe": (
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        "moe_config",
        "inplace",
        "activation",
        "w1_scale",
        "w2_scale",
        "w1_zp",
        "w2_zp",
        "a1_scale",
        "a2_scale",
        "block_shape",
        "global_num_experts",
        "expert_map",
        "routed_scaling_factor",
        "use_weight_shuffle",
        "output_dtype",
        "gemm1_alpha",
        "gemm1_limit",
    ),
}
_activation_lock = threading.Lock()
_activated = False
_commit_failure = None
_committing_error_type = None


def _after_fork_child():
    global _activation_lock, _commit_failure
    _activation_lock = threading.Lock()
    if _committing_error_type is not None and _commit_failure is None:
        _commit_failure = (_committing_error_type,
                           "fork inherited incomplete DAS publication; use a fresh process")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


class DasExtensionError(RuntimeError):
    """Raised before an incompatible optimization can be activated."""


def _require_vllm_032_release_line(raw_version: str | None) -> None:
    try:
        parsed = Version(raw_version) if raw_version is not None else None
    except (InvalidVersion, TypeError):
        parsed = None
    if (
        parsed is None
        or parsed.epoch != 0
        or parsed.release[:2] != TARGET_VLLM_RELEASE[:2]
    ):
        raise DasExtensionError(
            "DAS adapter extension requires vLLM >=0.32.0,<0.33.0; "
            f"received {raw_version!r}"
        )


def _require_callable(owner: object, name: str, qualified_name: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise DasExtensionError(f"required callable is unavailable: {qualified_name}")


def _require_signature(owner: object, name: str, qualified_name: str) -> None:
    _require_callable(owner, name, qualified_name)
    value = getattr(owner, name)
    try:
        actual = tuple(inspect.signature(value).parameters)
    except (TypeError, ValueError) as error:
        raise DasExtensionError(
            f"cannot inspect required callable: {qualified_name}"
        ) from error
    expected = _AITER_SIGNATURES[name]
    if actual != expected:
        raise DasExtensionError(
            f"required callable signature changed for {qualified_name}: "
            f"expected {expected}, got {actual}"
        )


def _require_commit_failure_abi():
    """Require the public host protocol before any shared registration."""
    try:
        api = _import_module("hcu_adapter.extensions")
        error_type = getattr(api, "ExtensionCommitError", None)
        if not inspect.isclass(error_type) or not issubclass(error_type, RuntimeError):
            raise TypeError("missing ExtensionCommitError")
        inspect.signature(error_type).bind("publication failed", reports=())
        bridge = _import_module("hcu_adapter.bridges.vllm")
        health = getattr(bridge, "require_runtime_healthy", None)
        classifier = getattr(api, "is_fatal_extension_error", None)
        for function, arguments in ((health, ()), (classifier, (RuntimeError(),))):
            if not callable(function):
                raise TypeError("missing public health query or fatal classifier")
            implementations = (function,) if inspect.isroutine(function) else (function, function.__call__)
            if any(check(value) for value in implementations for check in (
                inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction,
            )):
                raise TypeError("public failure protocol must be synchronous")
            inspect.signature(function).bind(*arguments)
        return api, health
    except Exception as error:
        raise DasExtensionError(
            "DAS requires Adapter public ExtensionCommitError, "
            "is_fatal_extension_error and require_runtime_healthy ABI; "
            f"preflight failed: {error}"
        ) from error


def activate_extension(context: ExtensionContext) -> None:
    """Prepare known boundaries, then publish once without rollback claims."""

    global _activated, _commit_failure, _committing_error_type
    if _commit_failure is not None:
        error_type, message = _commit_failure
        raise error_type(message)
    _require_vllm_032_release_line(context.vllm_version)
    fatal, health = _require_commit_failure_abi()
    health()  # Never wait on the host gate while holding the owner lock.
    with _activation_lock:
        if _commit_failure is not None:
            error_type, message = _commit_failure
            raise error_type(message)
        if _activated:
            if _FLA_CAPABILITY in context.requested_capabilities:
                fla = _import_module("hcu_vllm_das.fla")
                policy = json.loads(fla.effective_compile_factor())
                _require_fla_mount(context, policy.get("route") == "boltops")
            return
        try:
            lightop_norm = _import_module("lightop.norm")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required LightOp norm module: {error}"
            ) from error
        _require_callable(
            lightop_norm,
            "rmsnorm_forward_autograd",
            "lightop.norm.rmsnorm_forward_autograd",
        )
        _require_callable(
            lightop_norm,
            "fused_add_rms_norm",
            "lightop.norm.fused_add_rms_norm",
        )

        try:
            aiter_moe = _import_module("aiter.moe")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required AITER MoE module: {error}"
            ) from error
        moe_quant_type = getattr(aiter_moe, "MoeQuantType", None)
        if getattr(moe_quant_type, "W8A8", None) != "int8_w8a8":
            raise DasExtensionError(
                "required AITER quant type is unavailable or changed: "
                "aiter.moe.MoeQuantType.W8A8"
            )
        for name in _AITER_SIGNATURES:
            _require_signature(aiter_moe, name, f"aiter.moe.{name}")

        try:
            vllm_layernorm = _import_module(
                "vllm.model_executor.layers.layernorm"
            )
            torch_utils = _import_module("vllm.utils.torch_utils")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required vLLM 0.32 operator API: {error}"
            ) from error
        rms_norm = getattr(vllm_layernorm, "RMSNorm", None)
        _require_callable(
            rms_norm,
            "register_oot",
            "vllm.model_executor.layers.layernorm.RMSNorm.register_oot",
        )
        _require_callable(
            torch_utils,
            "direct_register_custom_op",
            "vllm.utils.torch_utils.direct_register_custom_op",
        )

        try:
            vllm_envs = _import_module("vllm.envs")
            fla = _import_module("hcu_vllm_das.fla")
            int8_oracle = _import_module("hcu_vllm_das.moe.int8_oracle")
            _require_callable(int8_oracle, "install", "hcu_vllm_das.moe.int8_oracle.install")
            # Pure checks before the first shared publication. These are not
            # reservations: later failures must still signal a dirty commit.
            from .environment import preflight_compile_factors
            preflight_compile_factors(vllm_envs)
            fla.preflight_compile_factors(vllm_envs)
            architecture = None
            mounted = False
            if fla.is_requested(vllm_envs):
                bridge = _import_module("hcu_adapter.bridges.vllm")
                detection = bridge.detect_hcu()
                if detection.detected:
                    architecture = detection.architecture
                    mounted = fla.preflight_install(vllm_envs, architecture=architecture)
            _require_fla_mount(context, mounted)
        except Exception as error:
            if fatal.is_fatal_extension_error(error):
                _commit_failure = (fatal.ExtensionCommitError,
                                   f"DAS preparation inherited a fatal failure: {error}")
                raise fatal.ExtensionCommitError(_commit_failure[1]) from error
            if isinstance(error, DasExtensionError):
                raise
            raise DasExtensionError(f"cannot prepare DAS operator capabilities: {error}") from error

        # Registering compile factors is already shared-state publication.
        # An exception from this point has uncertain effects; never continue
        # as a clean optional failure or silently retry it in this process.
        _committing_error_type = fatal.ExtensionCommitError
        try:
            register_compile_factors(vllm_envs)
            fla.register_compile_factors(vllm_envs)
            if mounted:
                mounted = fla.install(architecture=architecture)
            _require_fla_mount(context, mounted)
            _import_module("hcu_vllm_das.rms_norm")
            int8_oracle.install()
        except BaseException as error:
            message = f"DAS activation failed after publication: {type(error).__name__}: {error}"
            _commit_failure = (fatal.ExtensionCommitError, message)
            raise fatal.ExtensionCommitError(message) from error
        finally:
            _committing_error_type = None
        _activated = True


def _require_fla_mount(context: ExtensionContext, mounted: bool) -> None:
    if _FLA_CAPABILITY in context.requested_capabilities and not mounted:
        raise DasExtensionError(
            f"required capability {_FLA_CAPABILITY!r} is not mounted; "
            "enable the FLA flags and provide a supported HCU/BoltOPs ABI"
        )


def extension_spec() -> ExtensionSpec:
    """Return the adapter-owned discovery contract without runtime imports."""

    return ExtensionSpec(
        name="das-hcu",
        api_version=_IMPLEMENTED_EXTENSION_API_VERSION,
        priority=100,
        capabilities=_CAPABILITIES,
        activate=activate_extension,
    )


__all__ = [
    "DasExtensionError",
    "activate_extension",
    "extension_spec",
]
