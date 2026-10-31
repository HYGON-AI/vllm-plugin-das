# SPDX-License-Identifier: Apache-2.0
"""Independent declaration and preparation of the explicit HCU Flash backend.

This top-level module avoids attention.__init__'s backend import. Discovery
imports no framework or vendor. Activation checks public contracts;
ACTIVE does not establish a model's configuration, numerical or GPU support.
This owner does not install RMS, AITER, quantization aliases or lifecycle hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext, ExtensionSpec
from packaging.version import InvalidVersion, Version


CAPABILITY = "hcu-flash-attention"
BACKEND = "hcu_vllm_das.attention.flash_attn.HcuFlashAttentionBackend"
REGISTRY = "vllm.v1.attention.backends.registry"
_lock = RLock()
_published = None
_failure = None
_publishing = False


@dataclass(frozen=True)
class _Publication:
    pid: int
    enum: object
    register: object
    backend: type
    vendor: object


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited unfinished Flash registry publication; use a fresh spawned process"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


class FlashActivationError(RuntimeError):
    """The explicit Flash capability cannot be prepared on this host."""


def _fatal(reason, error=None):
    global _failure
    if _failure is None:
        # This write must precede arbitrary exception formatting and cleanup.
        # __str__ can raise BaseException, re-enter or fork after publication.
        _failure = reason
        if error is not None:
            try:
                detail = f"{reason}: {type(error).__name__}: {error}"
            except BaseException:
                detail = reason
            _failure = detail
    raise ExtensionCommitError(_failure) from error


def _owned(registry, publication):
    enum = registry.AttentionBackendEnum.FLASH_ATTN
    return (enum is publication.enum and registry.register_backend is publication.register
            and enum.is_overridden() and enum.get_path() == BACKEND
            and enum.get_class() is publication.backend)


def require_published(backend):
    """Read/check only; never activate, republish, roll back or clear failure."""
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _published is None:
            raise FlashActivationError("Flash registry publication is not active in this process")
        if _published.pid != os.getpid():
            raise FlashActivationError("Flash requires current-process runtime activation after fork")
        try:
            valid = (_owned(import_module(REGISTRY), _published)
                     and backend is _published.backend
                     and import_module("hcu_vllm_das.attention.flash_attn")._vendor_varlen_func() is _published.vendor)
        except BaseException as error:
            _fatal("published Flash ownership check failed", error)
        if not valid:
            _fatal("published Flash registry/class/provider state drift; no automatic recovery")


def check_health() -> None:
    """Host callback; read-only, no host imports/locks/activation or repair."""
    require_published(None if _published is None else _published.backend)


def activate(context: ExtensionContext) -> None:
    global _published, _publishing
    if _failure is not None:
        _fatal(_failure)
    try:
        version = Version(context.vllm_version or "")
    except InvalidVersion as error:
        if _published is not None:
            _fatal("published Flash activation version drift")
        raise FlashActivationError("HCU Flash requires the vLLM 0.32 release line") from error
    if version.epoch != 0 or version.release[:2] != (0, 32):
        if _published is not None:
            _fatal("published Flash activation version drift")
        raise FlashActivationError("HCU Flash requires the vLLM 0.32 release line")

    # Hardware/runtime admission belongs to Adapter. This owner prepares only
    # its backend/vendor ABI, then uses the official process-local registry.
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publishing:
            _fatal("recursive Flash registry publication")
        try:
            registry = import_module(REGISTRY)
            enum = registry.AttentionBackendEnum.FLASH_ATTN
        except BaseException as error:
            if _published is not None:
                _fatal("published Flash registry unavailable", error)
            raise
        if _published is not None:
            try:
                valid = (_owned(registry, _published)
                    and import_module("hcu_vllm_das.attention.flash_attn")._vendor_varlen_func() is _published.vendor)
            except BaseException as error:
                _fatal("published Flash state check failed", error)
            if not valid:
                _fatal("published Flash registry/class/provider state drift; no automatic recovery")
            # An inherited exact publication is usable only after this explicit
            # fresh runtime activation, never by trusting a parent ACTIVE report.
            _published = _Publication(os.getpid(), enum, _published.register,
                                      _published.backend, _published.vendor)
            return
        if enum.is_overridden():
            raise FlashActivationError("FLASH_ATTN already has a foreign registry override")
        try:
            backend = import_module("hcu_vllm_das.attention.flash_attn")
            cls = backend.HcuFlashAttentionBackend
            vendor = backend._vendor_varlen_func()
            if not isinstance(cls, type) or not callable(cls.validate_configuration) or not callable(vendor):
                raise TypeError("invalid backend class/validation/vendor callable")
            # Import-time Triton/native dependencies must fail before publishing.
            for module, name in (("e5m2_cache", "reshape_and_cache_e5m2"),):
                if not callable(getattr(import_module("hcu_vllm_das.attention."+module),name)):
                    raise TypeError("noncallable Flash dependency: "+name)
            for name in ("get_path", "get_class", "is_overridden"):
                if not callable(getattr(enum,name)):
                    raise TypeError("missing public registry method: "+name)
            if not callable(registry.register_backend):
                raise TypeError("missing public register_backend")
        except Exception as error:
            raise FlashActivationError(
                f"cannot prepare HCU Flash backend: {type(error).__name__}: {error}"
            ) from error
        if enum.is_overridden():
            raise FlashActivationError("FLASH_ATTN foreign override appeared during preparation")
        publication = _Publication(os.getpid(), enum, registry.register_backend, cls, vendor)
        _publishing = True
        try:
            registry.register_backend(enum, BACKEND)
            # Once publication was attempted, an exception has uncertain shared
            # effects. Signal publicly and latch; never advertise clean optional.
            _published = publication
            if not _owned(registry, publication):
                _fatal("Flash registry publication verification failed")
        except BaseException as error:
            _fatal("Flash registry commit failed", error)
        finally:
            _publishing = False


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-flash",
        api_version=1,
        priority=80,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
