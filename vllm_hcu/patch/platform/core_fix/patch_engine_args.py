# SPDX-License-Identifier: Apache-2.0
"""EngineArgs compatibility adapter for HCU-owned feature configuration.

The adapter intentionally does not add dataclass fields.  It removes legacy
HCU keyword arguments before the generated vLLM constructor sees them and
stores their normalized representation under ``additional_config['hcu']``.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass as official_dataclass
from types import ModuleType
from typing import Any

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config, set_hcu_config
from vllm_hcu.patch.runtime_state import PATCH_REGISTRY, PatchRegistry, run_patch

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.engine.arg_utils"
PATCH_ID = "platform.core_fix.hcu_config.engine_args"
TARGETS = (
    f"{TARGET_MODULE}.EngineArgs.__init__",
    f"{TARGET_MODULE}.AsyncEngineArgs.__init__",
    f"{TARGET_MODULE}.EngineArgs.from_cli_args",
    f"{TARGET_MODULE}.EngineArgs.create_engine_config",
    f"{TARGET_MODULE}.EngineArgs.add_cli_args",
)
_MARKER = "_vllm_hcu_feature_sidecar_patch_applied"
_DATACLASS_BRIDGE_MARKER = "_vllm_hcu_engine_args_dataclass_bridge"
_MISSING = object()
_HCU_BOOLEAN_KWARGS = (
    "enable_lightly_cp",
    "enable_lightly_cplb",
    "enable_custom_sp",
    "enable_multi_layers_mtp",
)
_DPSK_BACKEND = "dpsk_deep_gemm"
_UPSTREAM_BACKEND = "auto"
_DEEPEP_AUTO_BACKEND = "deepep_auto"
_DEEPEP_AUTO_UPSTREAM_BACKEND = "deepep_low_latency"
_FLASH_ATTN_BACKEND = "FLASH_ATTN"
_HCU_FLASH_ATTN_ALIASES = {
    "FLASH_ATTN_CLASSIC": "classic",
    "FLASH_ATTN_CUTLASS": "cutlass",
    "FLASH_ATTN_CUSTOM": "custom",
}


def _require_engine_args_class(module: ModuleType, name: str) -> type:
    value = getattr(module, name, None)
    if not isinstance(value, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.{name} is missing"
        )
    init = vars(value).get("__init__")
    if not callable(init):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.{name}.__init__ is missing"
        )
    try:
        parameters = inspect.signature(init).parameters
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {TARGET_MODULE}.{name}.__init__"
        ) from exc
    required = {
        "additional_config",
        "all2all_backend",
        "attention_backend",
        "attention_config",
        "moe_backend",
        "kernel_config",
        "speculative_config",
    }
    if not required.issubset(parameters):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.{name}.__init__ has "
            f"incompatible signature {inspect.signature(init)}"
        )
    return value


def _normalise_constructor_kwargs(
    signature: inspect.Signature,
    self: object,
    args: tuple[Any, ...],
    kwargs: MutableMapping[str, Any],
) -> HcuFeatureConfig:
    top_level_multi_mtp = kwargs.get("enable_multi_layers_mtp")
    top_level_multi_mtp_present = "enable_multi_layers_mtp" in kwargs
    updates = {
        name: kwargs.pop(name)
        for name in _HCU_BOOLEAN_KWARGS
        if name in kwargs
    }
    # Bind after removing HCU-only keywords so positional official arguments
    # (especially additional_config) participate in normalization instead of
    # being overwritten by a default sidecar after construction.
    try:
        bound = signature.bind_partial(self, *args, **kwargs)
    except TypeError:
        # Preserve the generated dataclass constructor's normal diagnostics.
        raise

    additional = bound.arguments.get("additional_config")
    if additional is not None and not isinstance(additional, Mapping):
        raise TypeError("additional_config must be a mapping or None")
    feature_config = get_hcu_config({"additional_config": additional})
    explicit_sidecar: Mapping[str, Any] = {}
    if isinstance(additional, Mapping):
        raw_hcu = additional.get("hcu")
        if isinstance(raw_hcu, HcuFeatureConfig):
            explicit_sidecar = raw_hcu.to_dict()
        elif isinstance(raw_hcu, Mapping):
            explicit_sidecar = raw_hcu
    for name, value in updates.items():
        if name in explicit_sidecar and explicit_sidecar[name] != value:
            raise ValueError(
                f"conflicting {name} values in HCU sidecar and top-level "
                "EngineArgs keyword"
            )

    requested_flash_mode: str | None = None
    top_level_attention_backend = bound.arguments.get("attention_backend")
    if isinstance(top_level_attention_backend, str):
        requested_flash_mode = _HCU_FLASH_ATTN_ALIASES.get(
            top_level_attention_backend.upper()
        )
        if requested_flash_mode is not None:
            if "attention_backend" not in kwargs:
                raise TypeError(
                    "positional HCU flash-attention aliases are not supported; "
                    "pass attention_backend by keyword"
                )
            kwargs["attention_backend"] = _FLASH_ATTN_BACKEND

    attention_config = bound.arguments.get("attention_config")
    if isinstance(attention_config, Mapping):
        nested_backend = attention_config.get("backend")
        if isinstance(nested_backend, str):
            nested_mode = _HCU_FLASH_ATTN_ALIASES.get(nested_backend.upper())
            if nested_mode is not None:
                if "attention_config" not in kwargs:
                    raise TypeError(
                        "positional attention_config containing an HCU "
                        "flash-attention alias is not supported; pass it by keyword"
                    )
                if requested_flash_mode is not None and requested_flash_mode != nested_mode:
                    raise ValueError(
                        "conflicting HCU flash-attention aliases in attention_backend "
                        "and attention_config.backend"
                    )
                requested_flash_mode = nested_mode
                normalized_attention = dict(attention_config)
                normalized_attention["backend"] = _FLASH_ATTN_BACKEND
                kwargs["attention_config"] = normalized_attention

    if requested_flash_mode is not None:
        existing_mode = feature_config.hcu_flash_attn_mode
        if existing_mode is not None and existing_mode != requested_flash_mode:
            raise ValueError(
                "conflicting hcu_flash_attn_mode in HCU sidecar and attention backend alias"
            )
        feature_config = feature_config.with_updates(
            hcu_flash_attn_mode=requested_flash_mode
        )

    requested_deepep_auto = feature_config.deepep_auto
    top_level_all2all = bound.arguments.get("all2all_backend")
    if top_level_all2all == _DEEPEP_AUTO_BACKEND:
        if "all2all_backend" not in kwargs:
            raise TypeError(
                "positional deepep_auto is not supported; pass "
                "all2all_backend='deepep_auto' so HCU can normalize it before "
                "upstream validation"
            )
        requested_deepep_auto = True
        kwargs["all2all_backend"] = _DEEPEP_AUTO_UPSTREAM_BACKEND
    elif requested_deepep_auto and top_level_all2all not in (
        None,
        _DEEPEP_AUTO_UPSTREAM_BACKEND,
    ):
        raise ValueError(
            "HCU sidecar selects deepep_auto but EngineArgs.all2all_backend "
            f"selects {top_level_all2all!r}"
        )

    requested_dpsk = feature_config.moe_backend == _DPSK_BACKEND
    top_level_backend = bound.arguments.get("moe_backend")
    if top_level_backend == _DPSK_BACKEND:
        if "moe_backend" not in kwargs:
            raise TypeError(
                "positional dpsk_deep_gemm is not supported; pass "
                "moe_backend='dpsk_deep_gemm' so HCU can normalize it before "
                "upstream validation"
            )
        requested_dpsk = True
        kwargs["moe_backend"] = _UPSTREAM_BACKEND
    elif requested_dpsk and top_level_backend not in (None, _UPSTREAM_BACKEND):
        raise ValueError(
            "HCU sidecar selects dpsk_deep_gemm but EngineArgs.moe_backend "
            f"selects {top_level_backend!r}"
        )

    kernel_config = bound.arguments.get("kernel_config")
    if isinstance(kernel_config, Mapping):
        nested_backend = kernel_config.get("moe_backend")
        if nested_backend == _DPSK_BACKEND:
            if "kernel_config" not in kwargs:
                raise TypeError(
                    "positional KernelConfig with dpsk_deep_gemm is not "
                    "supported; pass kernel_config by keyword"
                )
            requested_dpsk = True
            normalized_kernel = dict(kernel_config)
            normalized_kernel["moe_backend"] = _UPSTREAM_BACKEND
            kwargs["kernel_config"] = normalized_kernel
        elif requested_dpsk and nested_backend not in (None, _UPSTREAM_BACKEND):
            raise ValueError(
                "HCU sidecar selects dpsk_deep_gemm but "
                f"KernelConfig.moe_backend selects {nested_backend!r}"
            )

    speculative_config = bound.arguments.get("speculative_config")
    if isinstance(speculative_config, Mapping) and (
        "enable_multi_layers_mtp" in speculative_config
    ):
        if "speculative_config" not in kwargs:
            raise TypeError(
                "positional speculative_config containing enable_multi_layers_mtp "
                "is not supported; pass speculative_config by keyword"
            )
        nested_multi_mtp = speculative_config["enable_multi_layers_mtp"]
        if (
            "enable_multi_layers_mtp" in explicit_sidecar
            and explicit_sidecar["enable_multi_layers_mtp"] != nested_multi_mtp
        ):
            raise ValueError(
                "conflicting enable_multi_layers_mtp values in HCU sidecar "
                "and speculative_config"
            )
        if (
            top_level_multi_mtp_present
            and top_level_multi_mtp != nested_multi_mtp
        ):
            raise ValueError(
                "conflicting enable_multi_layers_mtp values in the top-level "
                "EngineArgs keyword and speculative_config"
            )
        normalized_speculative = dict(speculative_config)
        normalized_speculative.pop("enable_multi_layers_mtp")
        kwargs["speculative_config"] = normalized_speculative
        updates["enable_multi_layers_mtp"] = nested_multi_mtp

    if requested_dpsk:
        updates["moe_backend"] = _DPSK_BACKEND
    if requested_deepep_auto:
        updates["deepep_auto"] = True
    return feature_config.with_updates(**updates) if updates else feature_config


def _normalise_existing_engine_args(engine_args: object) -> HcuFeatureConfig:
    feature_config = get_hcu_config(engine_args)

    attention_backend = getattr(engine_args, "attention_backend", None)
    if isinstance(attention_backend, str):
        requested_flash_mode = _HCU_FLASH_ATTN_ALIASES.get(
            attention_backend.upper()
        )
        if requested_flash_mode is not None:
            if (
                feature_config.hcu_flash_attn_mode is not None
                and feature_config.hcu_flash_attn_mode != requested_flash_mode
            ):
                raise ValueError(
                    "conflicting hcu_flash_attn_mode in HCU sidecar and "
                    "attention backend alias"
                )
            setattr(engine_args, "attention_backend", _FLASH_ATTN_BACKEND)
            feature_config = feature_config.with_updates(
                hcu_flash_attn_mode=requested_flash_mode
            )

    attention_config = getattr(engine_args, "attention_config", None)
    direct_mode = getattr(attention_config, "hcu_flash_attn_mode", None)
    if direct_mode is not None:
        if (
            feature_config.hcu_flash_attn_mode is not None
            and feature_config.hcu_flash_attn_mode != direct_mode
        ):
            raise ValueError(
                "conflicting hcu_flash_attn_mode in HCU sidecar and AttentionConfig"
            )
        feature_config = feature_config.with_updates(
            hcu_flash_attn_mode=direct_mode
        )
    requested_dpsk = feature_config.moe_backend == _DPSK_BACKEND

    requested_deepep_auto = feature_config.deepep_auto
    all2all_backend = getattr(engine_args, "all2all_backend", None)
    if all2all_backend == _DEEPEP_AUTO_BACKEND:
        requested_deepep_auto = True
        setattr(
            engine_args,
            "all2all_backend",
            _DEEPEP_AUTO_UPSTREAM_BACKEND,
        )
    elif requested_deepep_auto and all2all_backend != _DEEPEP_AUTO_UPSTREAM_BACKEND:
        raise ValueError(
            "HCU sidecar selects deepep_auto but EngineArgs.all2all_backend "
            f"selects {all2all_backend!r}"
        )

    backend = getattr(engine_args, "moe_backend", _UPSTREAM_BACKEND)
    if backend == _DPSK_BACKEND:
        requested_dpsk = True
        setattr(engine_args, "moe_backend", _UPSTREAM_BACKEND)
    elif requested_dpsk and backend != _UPSTREAM_BACKEND:
        raise ValueError(
            "HCU sidecar selects dpsk_deep_gemm but EngineArgs.moe_backend "
            f"selects {backend!r}"
        )

    kernel_config = getattr(engine_args, "kernel_config", None)
    nested_backend = getattr(kernel_config, "moe_backend", _UPSTREAM_BACKEND)
    if nested_backend == _DPSK_BACKEND:
        requested_dpsk = True
        setattr(kernel_config, "moe_backend", _UPSTREAM_BACKEND)
    elif requested_dpsk and nested_backend != _UPSTREAM_BACKEND:
        raise ValueError(
            "HCU sidecar selects dpsk_deep_gemm but KernelConfig.moe_backend "
            f"selects {nested_backend!r}"
        )

    if requested_dpsk and feature_config.moe_backend != _DPSK_BACKEND:
        feature_config = feature_config.with_updates(moe_backend=_DPSK_BACKEND)
    if requested_deepep_auto and not feature_config.deepep_auto:
        feature_config = feature_config.with_updates(deepep_auto=True)

    speculative_config = getattr(engine_args, "speculative_config", None)
    if isinstance(speculative_config, Mapping) and (
        "enable_multi_layers_mtp" in speculative_config
    ):
        nested_multi_mtp = speculative_config["enable_multi_layers_mtp"]
        if feature_config.enable_multi_layers_mtp != nested_multi_mtp:
            raise ValueError(
                "conflicting enable_multi_layers_mtp values in HCU sidecar "
                "and speculative_config"
            )
        feature_config = feature_config.with_updates(
            enable_multi_layers_mtp=nested_multi_mtp
        )
        normalized_speculative = dict(speculative_config)
        normalized_speculative.pop("enable_multi_layers_mtp")
        setattr(engine_args, "speculative_config", normalized_speculative)
    set_hcu_config(engine_args, feature_config)
    return feature_config


def _wrap_constructor(owner: type) -> None:
    original = vars(owner)["__init__"]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def hcu_init(self, *args: Any, **kwargs: Any) -> None:
        feature_config = _normalise_constructor_kwargs(
            signature,
            self,
            args,
            kwargs,
        )
        original(self, *args, **kwargs)
        # The upstream generated constructor has now initialized
        # additional_config.  Store a plain dict for hash/pickle/JSON support.
        set_hcu_config(self, feature_config)

    setattr(owner, "_vllm_hcu_original_init", original)
    setattr(owner, "__init__", hcu_init)


def apply_to_module(module: ModuleType) -> bool:
    """Install constructor/config wrappers on the exact audited target module."""

    arg_utils = load_exact_module(TARGET_MODULE, module)
    if getattr(arg_utils, _MARKER, False):
        return False

    engine_args = _require_engine_args_class(arg_utils, "EngineArgs")
    async_engine_args = _require_engine_args_class(arg_utils, "AsyncEngineArgs")
    create_engine_config = vars(engine_args).get("create_engine_config")
    from_cli_descriptor = vars(engine_args).get("from_cli_args")
    add_cli_descriptor = vars(engine_args).get("add_cli_args")
    if not callable(create_engine_config):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            "EngineArgs.create_engine_config is missing"
        )
    if not isinstance(from_cli_descriptor, classmethod):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            "EngineArgs.from_cli_args must be a classmethod"
        )
    if not isinstance(add_cli_descriptor, staticmethod):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            "EngineArgs.add_cli_args must be a staticmethod"
        )
    from_cli_args = from_cli_descriptor.__func__
    from_cli_signature = inspect.signature(from_cli_args)
    if tuple(from_cli_signature.parameters) != ("cls", "args"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            f"EngineArgs.from_cli_args has incompatible signature {from_cli_signature}"
        )
    signature = inspect.signature(create_engine_config)
    if tuple(signature.parameters) != ("self", "usage_context", "headless"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            f"EngineArgs.create_engine_config has incompatible signature {signature}"
        )
    add_cli_args = add_cli_descriptor.__func__
    add_cli_signature = inspect.signature(add_cli_args)
    if tuple(add_cli_signature.parameters) != ("parser",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[4]} has incompatible "
            f"signature {add_cli_signature}"
        )

    _wrap_constructor(engine_args)
    _wrap_constructor(async_engine_args)

    @classmethod
    @functools.wraps(from_cli_args)
    def hcu_from_cli_args(cls, args):
        engine_args_instance = from_cli_args(cls, args)
        feature_config = get_hcu_config(engine_args_instance)
        updates = {
            name: getattr(args, name)
            for name in _HCU_BOOLEAN_KWARGS[:3]
            if hasattr(args, name)
        }
        if updates:
            feature_config = feature_config.with_updates(**updates)
            set_hcu_config(engine_args_instance, feature_config)
        return engine_args_instance

    setattr(engine_args, "_vllm_hcu_original_from_cli_args", from_cli_descriptor)
    setattr(engine_args, "from_cli_args", hcu_from_cli_args)

    @functools.wraps(add_cli_args)
    def hcu_add_cli_args(parser):
        result = add_cli_args(parser)
        for action in getattr(result, "_actions", ()):
            if getattr(action, "dest", None) != "all2all_backend":
                continue
            choices = getattr(action, "choices", None)
            if choices is None:
                raise PatchCompatibilityError(
                    "--all2all-backend CLI action has no audited choices"
                )
            if _DEEPEP_AUTO_BACKEND not in choices:
                action.choices = tuple(choices) + (_DEEPEP_AUTO_BACKEND,)
            break
        else:
            raise PatchCompatibilityError(
                "EngineArgs.add_cli_args did not install --all2all-backend"
            )
        return result

    setattr(engine_args, "_vllm_hcu_original_add_cli_args", add_cli_descriptor)
    setattr(engine_args, "add_cli_args", staticmethod(hcu_add_cli_args))

    @functools.wraps(create_engine_config)
    def hcu_create_engine_config(self, *args: Any, **kwargs: Any):
        feature_config = _normalise_existing_engine_args(self)
        config = create_engine_config(self, *args, **kwargs)
        set_hcu_config(config, feature_config)
        if get_hcu_config(config) != feature_config:
            raise PatchCompatibilityError(
                "VllmConfig did not retain the normalized HCU feature sidecar"
            )
        return config

    setattr(engine_args, "_vllm_hcu_original_create_engine_config", create_engine_config)
    setattr(engine_args, "create_engine_config", hcu_create_engine_config)
    setattr(arg_utils, _MARKER, True)
    return True


def _run_bridge_patch(module: ModuleType, registry: PatchRegistry) -> bool:
    """Apply at the AsyncEngineArgs decorator boundary and verify the marker."""

    def install_and_verify() -> bool:
        result = apply_to_module(module)
        if not getattr(module, _MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch {PATCH_ID!r} did not install its target marker"
            )
        return result

    applied = run_patch(
        PATCH_ID,
        TARGETS,
        install_and_verify,
        registry=registry,
    )
    if not getattr(module, _MARKER, False):
        raise PatchCompatibilityError(
            f"required HCU patch {PATCH_ID!r} did not install its target marker"
        )
    return applied is not None


def arm_partial_import_bridge(
    module: ModuleType | None = None,
    *,
    registry: PatchRegistry = PATCH_REGISTRY,
) -> bool:
    """Bridge recursive platform discovery inside ``vllm.engine.arg_utils``.

    vLLM imports ``current_platform`` before defining ``EngineArgs``.  When
    platform discovery recursively installs HCU from that import, the normal
    exact post-import callback correctly remains armed, but there may be no
    later plugin boundary before a caller constructs ``EngineArgs``.  For this
    one exact audited target module, temporarily wrap its already-imported ``dataclass``
    global.  The wrapper applies the public patch transaction after
    ``AsyncEngineArgs`` has been decorated (so both generated constructors
    exist) and before the import statement publishes the completed module.

    No global import or dataclasses state is changed.  The module global is
    restored to the official decorator before control returns from the final
    class decorator, including on a latched compatibility failure.
    """

    if module is None:
        candidate = sys.modules.get(TARGET_MODULE)
        if candidate is None:
            return False
        if not isinstance(candidate, ModuleType):
            raise PatchCompatibilityError(
                f"required HCU patch expected module {TARGET_MODULE!r}, "
                f"got {type(candidate).__name__}"
            )
        module = candidate
    module = load_exact_module(TARGET_MODULE, module)

    engine_args = getattr(module, "EngineArgs", None)
    async_engine_args = getattr(module, "AsyncEngineArgs", None)
    if isinstance(engine_args, type) and isinstance(async_engine_args, type):
        current_dataclass = getattr(module, "dataclass", None)
        original = getattr(current_dataclass, _DATACLASS_BRIDGE_MARKER, None)
        if original is not None:
            setattr(module, "dataclass", original)
        return _run_bridge_patch(module, registry)

    spec = getattr(module, "__spec__", None)
    if not bool(getattr(spec, "_initializing", False)):
        error = PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE!r} is only partially "
            "defined after module initialization completed"
        )

        def fail_incomplete_target() -> None:
            raise error

        run_patch(PATCH_ID, TARGETS, fail_incomplete_target, registry=registry)
        raise AssertionError("run_patch must propagate the compatibility error")

    current_dataclass = getattr(module, "dataclass", None)
    previous_original = getattr(current_dataclass, _DATACLASS_BRIDGE_MARKER, None)
    if previous_original is not None:
        # A recursive platform/general-plugin entry has already armed this
        # exact module object.  Keep the first bridge and its captured official
        # decorator authoritative.
        return False
    if current_dataclass is not official_dataclass:
        error = PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.dataclass does not "
            "reference the official dataclasses.dataclass decorator"
        )

        def fail_unexpected_decorator() -> None:
            raise error

        run_patch(PATCH_ID, TARGETS, fail_unexpected_decorator, registry=registry)
        raise AssertionError("run_patch must propagate the compatibility error")

    original_dataclass = current_dataclass

    def after_dataclass(candidate: type) -> type:
        if (
            not isinstance(candidate, type)
            or candidate.__module__ != TARGET_MODULE
            or candidate.__name__ != "AsyncEngineArgs"
        ):
            return candidate

        previous_async = module.__dict__.get("AsyncEngineArgs", _MISSING)
        module.AsyncEngineArgs = candidate
        try:
            _run_bridge_patch(module, registry)
        finally:
            # The wrapper is deliberately one-shot and exact.  Restore the
            # official module global even when application fails and latches.
            module.dataclass = original_dataclass
            if previous_async is _MISSING:
                module.__dict__.pop("AsyncEngineArgs", None)
            else:
                module.AsyncEngineArgs = previous_async
        return candidate

    @functools.wraps(original_dataclass)
    def hcu_dataclass(cls=None, /, **kwargs):
        if cls is not None:
            return after_dataclass(original_dataclass(cls, **kwargs))

        decorator = original_dataclass(**kwargs)

        @functools.wraps(decorator)
        def decorate(candidate):
            return after_dataclass(decorator(candidate))

        return decorate

    setattr(hcu_dataclass, _DATACLASS_BRIDGE_MARKER, original_dataclass)
    module.dataclass = hcu_dataclass
    return True


def apply(module: ModuleType | None = None) -> bool:
    arg_utils = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=arg_utils,
        marker=_MARKER,
        callback=lambda: apply_to_module(arg_utils),
    )


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "arm_partial_import_bridge",
]
