# SPDX-License-Identifier: Apache-2.0
"""Explicit, lazy publication of Qwen4 dense checkpoint-loader adapters."""
from __future__ import annotations

import os

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from packaging.version import InvalidVersion, Version

from .adapter_api import adapter_module

ENTRY = "das_hcu_qwen4_dense"
CAPABILITY = "qwen4-dense-qsa-checkpoint"
_ACTION = "qwen4-dense-model-publication"
_HEALTH_ACTION = "qwen4-dense-model-health"
_Lifecycle = adapter_module("runtime_lifecycle").ProcessLocalLifecycle
_lifecycle = _Lifecycle(runtime_action=_ACTION, error_type=ExtensionCommitError)
# Successful run_once publication cannot latch a later ownership failure.
# Use the same Adapter mechanism for that independent, failure-only action.
_health = _Lifecycle(runtime_action=_HEALTH_ACTION, error_type=ExtensionCommitError)
_publication = None
_ARCHES = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP")
_MODULE = "hcu_vllm_das.models.qwen4_dense"


def _key(entry):
    return getattr(entry, "module_name", None), getattr(entry, "class_name", None)


def _reject(reason):
    def fail():
        raise ExtensionCommitError(reason)

    _health.run_once(_HEALTH_ACTION, fail)


def _is_hcu():
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and current_platform.device_name == "hcu"


def _owned():
    from vllm.model_executor.models import ModelRegistry

    _, registry, entries = _publication
    return registry is ModelRegistry and all(
        registry.models.get(arch) is entry
        and _key(entry) == (_MODULE, "Hcu" + arch)
        for arch, entry in entries.items()
    )


def check_health():
    """Read ownership only. A failed publication or drift is never repaired."""
    _lifecycle.require_runtime_healthy()
    _health.require_runtime_healthy()
    if _publication is None:
        raise RuntimeError("Qwen4 dense checkpoint extension has not activated")
    try:
        valid = _owned()
    except BaseException:
        _reject("Qwen4 dense model ownership check failed")
    if not valid:
        _reject("Qwen4 dense model ownership changed; restart process")
    if _publication[0] != os.getpid():
        raise RuntimeError("Qwen4 dense requires current-process activation after fork")


def activate(context):
    _lifecycle.require_runtime_healthy()
    _health.require_runtime_healthy()
    try:
        version = Version(context.vllm_version or "")
    except InvalidVersion:
        version = None
    requested = (ENTRY in context.requested_extensions
                 or CAPABILITY in context.requested_capabilities)
    if (not requested or version is None or version.epoch != 0
            or version.release[:2] != (0, 33)):
        if _publication is not None:
            _reject("Qwen4 dense activation policy changed; restart process")
        raise RuntimeError("Qwen4 dense requires explicit selection and vLLM 0.33")
    if not _is_hcu():
        if _publication is not None:
            _reject("Qwen4 dense HCU platform changed; restart process")
        raise RuntimeError("Qwen4 dense checkpoint extension requires HCU")

    def publish():
        global _publication
        try:
            from vllm.model_executor.models import ModelRegistry

            if _publication is not None:
                # Healthy fork: registry records survive, lifecycle locks reset.
                if not _owned():
                    raise ExtensionCommitError("inherited Qwen4 dense ownership changed")
                _publication = (os.getpid(), *_publication[1:])
                return
            # Validate every owner before the first public registry mutation.
            if any(_key(ModelRegistry.models.get(arch))
                   != ("vllm.models.qwen4_exp", arch) for arch in _ARCHES):
                raise RuntimeError("Qwen4 model has a foreign or missing owner")
            entries = {}
            for arch in _ARCHES:
                ModelRegistry.register_model(arch, _MODULE + ":Hcu" + arch)
                entry = ModelRegistry.models.get(arch)
                if _key(entry) != (_MODULE, "Hcu" + arch):
                    raise ExtensionCommitError("Qwen4 dense publication lost ownership")
                entries[arch] = entry
            _publication = (os.getpid(), ModelRegistry, entries)
            if not _owned():
                raise ExtensionCommitError("Qwen4 dense publication verification failed")
        except BaseException as error:
            # Fixed diagnostics also cover BaseException and hostile __str__.
            # A partial registry update is retained and fatal, never rolled back.
            raise ExtensionCommitError("Qwen4 dense model publication failed") from error

    _lifecycle.run_once(_ACTION, publish)
    check_health()


def extension_spec():
    return ExtensionSpec(name=ENTRY, api_version=1, priority=75,
                         capabilities=frozenset({CAPABILITY}),
                         activate=activate, check_health=check_health)
