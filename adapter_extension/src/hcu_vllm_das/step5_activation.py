# SPDX-License-Identifier: Apache-2.0
"""Register standalone Step5 draft configs through public Transformers APIs.

Discovery imports no framework. Adapter calls activation before model-config
construction; its platform/general entry points remain the sole host owners.
This capability describes configuration parsing, not model execution.
"""

from __future__ import annotations

from importlib import import_module
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext, ExtensionSpec
from packaging.version import InvalidVersion, Version


CAPABILITY = "step5-standalone-mtp-config"
_lock = RLock()
_publication = None
_failure = None
_publishing = False


class Step5ConfigError(RuntimeError):
    """The standalone configuration capability cannot be prepared."""


def _fatal(reason):
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited unfinished Step5 config publication; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _owned():
    _, mapping, cls = _publication
    return (
        mapping["step5_mtp"] is cls
        and import_module("transformers").CONFIG_MAPPING is mapping
        and import_module("hcu_vllm_das.step5_config").Step5MTPConfig is cls
        and import_module("vllm.transformers_utils.configs").Step5Config.model_type
        == "step5"
    )


def check_health():
    """Read-only ownership check; never re-register a changed mapping."""
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publication is None:
            raise Step5ConfigError("Step5 config extension has not activated")
        try:
            valid = _owned()
        except BaseException:
            _fatal("published Step5 config ownership check failed")
        if _failure is not None:
            _fatal(_failure)
        if not valid:
            _fatal("published Step5 config ownership drift; no automatic recovery")
        if _publication[0] != os.getpid():
            raise Step5ConfigError(
                "Step5 config requires current-process activation after fork"
            )


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
            _fatal("published Step5 config release-line drift")
        raise Step5ConfigError(
            "Step5 config requires the epoch-zero vLLM 0.33 release line"
        )
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publishing:
            _fatal("recursive Step5 config publication")
        _publishing = True
        try:
            if _publication is not None:
                try:
                    valid = _owned()
                except BaseException:
                    _fatal("published Step5 config ownership check failed")
                if not valid:
                    _fatal(
                        "published Step5 config ownership drift; no automatic recovery"
                    )
                _publication = (os.getpid(), *_publication[1:])
                check_health()
                return
            transformers = import_module("transformers")
            cls = import_module("hcu_vllm_das.step5_config").Step5MTPConfig
            mapping = transformers.CONFIG_MAPPING
            # AutoConfig.register(exist_ok=False) alone does not protect an
            # existing out-of-tree entry in Transformers' extra-content map.
            if "step5_mtp" in mapping:
                raise Step5ConfigError("step5_mtp already has a config owner")
            if _failure is not None:
                _fatal(_failure)
            try:
                transformers.AutoConfig.register("step5_mtp", cls)
                _publication = (os.getpid(), mapping, cls)
                check_health()
            except BaseException:
                _fatal("Step5 config registration commit failed; use a fresh process")
        finally:
            _publishing = False


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-step5-config",
        api_version=1,
        priority=70,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
