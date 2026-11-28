# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model-scoped publication of the active Core033 Flash implementation.

The registered Flash config requests this runtime when parsed or restored in a
worker. Explicit required capabilities and extensions still assert startup
availability. Unselected discovery imports no model.
ACTIVE certifies owner/import compatibility only, never checkpoint, HIP,
DFly generation, HiSparse or performance acceptance.
"""

from dataclasses import dataclass
from importlib import import_module
import inspect
import os
from threading import RLock

from hcu_adapter.extensions import (
    ExtensionCommitError,
    ExtensionSpec,
    is_fatal_extension_error,
)
from packaging.version import InvalidVersion, Version

from .model_runtime_activation import ModelRuntimeActivation

CAPABILITY = "hyv4-flash-model"
_ENTRY_POINT = "das_hcu_hyv4_flash_model"
_SPEC_NAME = "das-hcu-hyv4-flash-model"
_ARCH = "HYV4FlashForCausalLM"
_MODULE = "hcu_vllm_das.models.hy_v4_flash.model"
_CLASS = "HYV4FlashForCausalLM"
_VL_ARCH = "HYV4FlashVLForConditionalGeneration"
_VL_MODULE = "hcu_vllm_das.models.hy_v4_flash.vl_model"
_VL_CLASS = _VL_ARCH
_CONFIG_MODULE = "hcu_vllm_das.hyv4_flash_config_activation"
_COMPAT_MODULE = "hcu_vllm_das.models.hy_v4_flash.dfly_compat"
_RUNNER_MODULE = "hcu_vllm_das.models.hy_v4_flash.fullgraph_runner"
_lock = RLock()
_publication = None
_publishing = False
_checking = False
_failure = None
_selected = None


@dataclass(frozen=True)
class _Publication:
    pid: int
    selection: tuple[bool, bool]
    registry: object
    entry: object
    model_cls: type
    register: tuple[object, object]
    config: object
    config_activate: object
    config_health: object
    compat: object
    compat_install: object
    compat_health: object
    vl_entry: object
    vl_model_cls: type
    runner: object
    runner_install: object
    runner_health: object
    runner_public_health: object


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _raise_if_failed():
    if _failure is not None:
        _fatal(_failure)


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if (_publication is not None or _publishing or _checking) and _failure is None:
        _failure = "Flash runtime publication inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _selection(extensions, capabilities):
    # API1 matches required extensions by entry_point_name, NOT spec.name.
    # Do not publish and only then let the host reject an impossible selector.
    if (_SPEC_NAME in extensions or _SPEC_NAME in capabilities
            or CAPABILITY in extensions or _ENTRY_POINT in capabilities):
        raise RuntimeError(
            "Flash selection must use entry-point name " + _ENTRY_POINT
            + " in required extensions, or " + CAPABILITY + " in capabilities"
        )
    return _ENTRY_POINT in extensions, CAPABILITY in capabilities


def _environment_selection():
    def names(key):
        return {item.strip() for item in os.environ.get(key, "").split(",")
                if item.strip()}
    return _selection(names("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS"),
                      names("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES"))


def _selection_error(message):
    if _publication is not None:
        _fatal(message)
    raise RuntimeError(message)


def _registrar_identity(method):
    return getattr(method, "__self__", None), getattr(method, "__func__", method)


def _same_registrar(method, expected):
    current = _registrar_identity(method)
    return current[0] is expected[0] and current[1] is expected[1]


def _require_sync(function, *args):
    if not callable(function):
        raise RuntimeError("Flash lifecycle provider must be callable")
    predicates = (inspect.iscoroutinefunction, inspect.isasyncgenfunction,
                  inspect.isgeneratorfunction)
    if any(test(value) for test in predicates
           for value in (function, getattr(function, "__call__", None))):
        raise RuntimeError("Flash lifecycle providers must be synchronous")
    try:
        inspect.signature(function).bind(*args)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Flash lifecycle provider signature is incompatible"
        ) from error


def _invoke_sync(function, *args):
    result = function(*args)
    if (inspect.isawaitable(result) or inspect.isgenerator(result)
            or inspect.isasyncgen(result)):
        if inspect.iscoroutine(result):
            result.close()
        raise RuntimeError("Flash lifecycle provider must complete synchronously")
    return result


def _owned(publication):
    registry = import_module("vllm.model_executor.models").ModelRegistry
    config = import_module(_CONFIG_MODULE)
    compat = import_module(_COMPAT_MODULE)
    runner = import_module(_RUNNER_MODULE)
    return (
        publication.pid == os.getpid()
        and registry is publication.registry
        and registry.models.get(_ARCH) is publication.entry
        and publication.entry.module_name == _MODULE
        and publication.entry.class_name == _CLASS
        and _same_registrar(registry.register_model, publication.register)
        and getattr(import_module(_MODULE), _CLASS) is publication.model_cls
        and registry.models.get(_VL_ARCH) is publication.vl_entry
        and publication.vl_entry.module_name == _VL_MODULE
        and publication.vl_entry.class_name == _VL_CLASS
        and getattr(import_module(_VL_MODULE), _VL_CLASS) is publication.vl_model_cls
        and config is publication.config
        and config.activate is publication.config_activate
        and config.check_health is publication.config_health
        and compat is publication.compat
        and compat.install_compat is publication.compat_install
        and compat.check_health is publication.compat_health
        and runner is publication.runner
        and runner.install is publication.runner_install
        and runner._check_registration is publication.runner_health
        and runner.check_health is publication.runner_public_health
    )


def check_health():
    """Read-only checks; never repair drift, retry commits or clear a latch."""
    global _checking
    with _lock:
        _raise_if_failed()
        if _checking or (_publishing and _publication is None):
            _fatal("recursive Flash model health check")
        if _publication is None:
            if _selected is False and not any(_environment_selection()):
                return
            raise RuntimeError("Flash model extension has not activated")
        _checking = True
        try:
            if _environment_selection() != _publication.selection:
                _fatal("Flash runtime selection changed after publication")
            if not _owned(_publication):
                _fatal("Flash model/config/DFly ownership drift; use a fresh process")
            _raise_if_failed()
            _invoke_sync(_publication.config_health)
            _raise_if_failed()
            _invoke_sync(_publication.compat_health)
            _raise_if_failed()
            _invoke_sync(_publication.runner_health)
            _raise_if_failed()
            # Providers may run callbacks: verify again and notice swallowed
            # recursive failures instead of reporting a healthy publication.
            if not _owned(_publication):
                _fatal("Flash model/config/DFly ownership drift; use a fresh process")
            _raise_if_failed()
        except BaseException:
            _fatal("Flash model/config/DFly runtime ownership check failed")
        finally:
            _checking = False


def _activate(context, expected_selection=None, *, model_selected=False):
    global _publication, _publishing, _selected
    with _lock:
        _raise_if_failed()
        if _publishing or _checking:
            _fatal("recursive Flash model publication or activation during health")
        try:
            selection = _environment_selection()
            requested = _selection(context.requested_extensions,
                                   context.requested_capabilities)
        except RuntimeError as error:
            _selection_error(str(error))
        if selection != requested or (
            expected_selection is not None and selection != expected_selection
        ):
            _selection_error(
                "Flash runtime context, discovery and public required-selection "
                "environment must agree; select "
                "HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=" + CAPABILITY
            )
        if not any(selection) and not model_selected:
            if _publication is not None:
                _fatal("Flash runtime cannot be disabled after publication")
            _selected = False
            return
        try:
            version = Version(context.vllm_version or "0")
        except InvalidVersion:
            version = None
        if (version is None or version.epoch != 0
                or version.release != (0, 33, 0) or version.is_prerelease
                or version.is_devrelease or version.post is not None):
            message = "HYV4 Flash runtime requires pinned vLLM 0.33.0"
            if _publication is not None:
                _fatal(message + "; published runtime version drift")
            raise RuntimeError(message)
        _selected = True
        if _publication is not None:
            if selection != _publication.selection:
                _fatal("Flash runtime selection changed after publication")
            check_health()
            return
        _publishing = True
        try:
            config = import_module(_CONFIG_MODULE)
            config_activate, config_health = config.activate, config.check_health
            _require_sync(config_activate, context)
            _require_sync(config_health)
            _invoke_sync(config_activate, context)
            registry = import_module("vllm.model_executor.models").ModelRegistry
            if any(arch in registry.models for arch in (_ARCH, _VL_ARCH)):
                raise RuntimeError("HYV4 Flash already has a model registry owner")
            for leaf in ("model", "attention", "sparse_backend", "swa",
                         "compressed_indexer", "indexpool_cache", "engram", "dfly"):
                import_module("hcu_vllm_das.models.hy_v4_flash." + leaf)
            model_cls = getattr(import_module(_MODULE), _CLASS)
            vl_model_cls = getattr(import_module(_VL_MODULE), _VL_CLASS)
            torch = import_module("torch")
            if any(not isinstance(cls, type) or not issubclass(cls, torch.nn.Module)
                   for cls in (model_cls, vl_model_cls)):
                raise RuntimeError("Flash registry target is not an actual model class")
            compat = import_module(_COMPAT_MODULE)
            install, health = compat.install_compat, compat.check_health
            _require_sync(install)
            _require_sync(health)
            runner = import_module(_RUNNER_MODULE)
            runner_install = runner.install
            runner_health = runner._check_registration
            runner_public_health = runner.check_health
            _require_sync(runner_install)
            _require_sync(runner_health)
            _require_sync(runner_public_health)
            _invoke_sync(config_health)
            _raise_if_failed()
            if any(arch in registry.models for arch in (_ARCH, _VL_ARCH)):
                raise RuntimeError("HYV4 Flash already has a model registry owner")
            register = _registrar_identity(registry.register_model)
            try:
                # Calling install crosses a potentially irreversible boundary.
                # The DFly owner supplies the real missing Core033 startup seams.
                _invoke_sync(install)
                _raise_if_failed()
                if os.environ.get("HCU_VLLM_HY4_HISPARSE") == "1":
                    for leaf in ("hisparse_config_compat", "hisparse_planner_compat"):
                        provider = import_module("hcu_vllm_das.models.hy_v4_flash." + leaf)
                        _require_sync(provider.install_compat)
                        _invoke_sync(provider.install_compat)
                _invoke_sync(runner_install)
                _raise_if_failed()
                current_registry = import_module(
                    "vllm.model_executor.models"
                ).ModelRegistry
                if (current_registry is not registry
                        or any(arch in registry.models for arch in (_ARCH, _VL_ARCH))
                        or not _same_registrar(registry.register_model, register)):
                    _fatal("Flash model registry changed during DFly publication")
                registry.register_model(_ARCH, f"{_MODULE}:{_CLASS}")
                registry.register_model(_VL_ARCH, f"{_VL_MODULE}:{_VL_CLASS}")
                _publication = _Publication(
                    os.getpid(), selection, registry, registry.models[_ARCH], model_cls,
                    register, config, config_activate, config_health,
                    compat, install, health, registry.models[_VL_ARCH], vl_model_cls,
                    runner, runner_install, runner_health,
                    runner_public_health,
                )
                check_health()
            except BaseException:
                _fatal("Flash runtime publication failed; restart a clean process")
        except BaseException as error:
            _raise_if_failed()
            if is_fatal_extension_error(error):
                _fatal(
                    "Flash runtime preflight dependency commit failed; "
                    "use a fresh process"
                )
            raise
        finally:
            _publishing = False


def activate(context):
    """Direct activation still requires the same explicit environment selection."""
    _activate(context)


def _activate_for_model(context):
    _activate(context, model_selected=True)


_model_activation = ModelRuntimeActivation(_activate_for_model, spawn_by_default=True)


def request_model_runtime():
    _model_activation.request()


def _requested_by_environment():
    return any(_environment_selection())


def extension_spec():
    # Freeze discovery selection: the host reports spec.capabilities verbatim
    # and does NOT automatically execute the health checker after activate().
    with _lock:
        _raise_if_failed()
        selection = _environment_selection()

    def activate_discovered(context):
        model_selected = _model_activation.requested
        try:
            _activate(context, selection, model_selected=model_selected)
            _model_activation.bind(context)
        except Exception as error:
            if (_model_activation.requested and not any(selection)
                    and not is_fatal_extension_error(error)):
                # This process already received a Flash model configuration.
                # The usual optional-extension recovery cannot serve that model.
                _fatal(f"selected Flash runtime activation failed: {error}")
            raise

    def health_discovered():
        with _lock:
            _raise_if_failed()
            try:
                current = _environment_selection()
            except RuntimeError as error:
                _selection_error(str(error))
            if current != selection:
                _selection_error("Flash runtime discovery selection changed")
            check_health()

    return ExtensionSpec(
        name=_SPEC_NAME, api_version=1, priority=72,
        capabilities=frozenset({CAPABILITY}) if any(selection) else frozenset(),
        activate=activate_discovered,
        check_health=health_discovered,
    )
