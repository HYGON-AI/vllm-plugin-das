# SPDX-License-Identifier: Apache-2.0
"""Publish Flash config types using the Adapter lifecycle and public AutoConfig.

This capability parses metadata only. It does not register model implementations
or claim that Flash inference is available in the active package.
"""

from importlib import import_module
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext, ExtensionSpec
from packaging.version import InvalidVersion, Version

CAPABILITY = "hyv4-flash-config"
_CONFIG_MODULE = "hcu_vllm_das.models.hy_v4_flash.configs"
_TYPES = (("hy_v4_flash", "HYV4FlashConfig"),
          ("hy_v4_flash_vl", "HYV4FlashVLConfig"))
_lock = RLock()
_publication = None
_failure = None
_publishing = False


class FlashConfigError(RuntimeError):
    """The metadata extension cannot be prepared on the current host."""


def _fatal(reason):
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited unfinished Flash config publication; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _owned():
    _, mapping, classes, core_class = _publication
    module = import_module(_CONFIG_MODULE)
    return (
        import_module("transformers").CONFIG_MAPPING is mapping
        and import_module("vllm.transformers_utils.configs.hy_v4").HYV4Config
        is core_class
        and core_class.model_type == "hy_v4"
        and all(mapping[key] is cls and cls.model_type == key
                and getattr(module, symbol) is cls
                for (key, symbol), cls in zip(_TYPES, classes))
    )


def check_health():
    """Check ownership without replacing foreign mappings or retrying commits."""
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publication is None:
            raise FlashConfigError("Flash config extension has not activated")
        try:
            valid = _owned()
        except BaseException:
            _fatal("published Flash config ownership check failed")
        if _failure is not None:
            _fatal(_failure)
        if not valid:
            _fatal("published Flash config ownership drift; no automatic recovery")
        if _publication[0] != os.getpid():
            raise FlashConfigError("Flash config requires current-process activation after fork")


def activate(context: ExtensionContext):
    global _publication, _publishing
    if _failure is not None:
        _fatal(_failure)
    try:
        version = Version(context.vllm_version or "")
    except InvalidVersion:
        version = None
    if version is None or version.epoch != 0 or version.release[:2] != (0, 33):
        if _publication is not None:
            _fatal("published Flash config release-line drift")
        raise FlashConfigError("Flash config requires the epoch-zero vLLM 0.33 release line")
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publishing:
            _fatal("recursive Flash config publication")
        _publishing = True
        try:
            if _publication is not None:
                try:
                    valid = _owned()
                except BaseException:
                    _fatal("published Flash config ownership check failed")
                if not valid:
                    _fatal("published Flash config ownership drift; no automatic recovery")
                _publication = (os.getpid(), *_publication[1:])
                check_health()
                return
            transformers = import_module("transformers")
            mapping = transformers.CONFIG_MAPPING
            # Preflight both names before the first irreversible registration.
            # AutoConfig.register alone may overwrite an out-of-tree extra entry.
            for key, _ in _TYPES:
                if key in mapping:
                    raise FlashConfigError(f"{key} already has a config owner")
            module = import_module(_CONFIG_MODULE)
            classes = tuple(getattr(module, symbol) for _, symbol in _TYPES)
            core_class = import_module("vllm.transformers_utils.configs.hy_v4").HYV4Config
            if core_class.model_type != "hy_v4":
                raise FlashConfigError("official HYV4 config identity is incompatible")
            if _failure is not None:
                _fatal(_failure)
            try:
                for (key, _), cls in zip(_TYPES, classes):
                    transformers.AutoConfig.register(key, cls)
                _publication = (os.getpid(), mapping, classes, core_class)
                check_health()
            except BaseException:
                _fatal("Flash config registration commit failed; use a fresh process")
        finally:
            _publishing = False


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-hyv4-flash-config",
        api_version=1,
        priority=71,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
