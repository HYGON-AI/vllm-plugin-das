# SPDX-License-Identifier: Apache-2.0
"""Explicit PCP producer runner publication through the Adapter factory seam."""

import importlib
import os
from threading import RLock

from hcu_adapter.bridges import vllm_model_runner as factory
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from packaging.version import Version

OWNER = "das_hcu_hy4_pcp_dfly"
_lock = RLock()
_registration = None
_owner = None
_failure = None


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if _registration is not None:
        _failure = "HY4 PCP runner inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def check_health():
    global _failure
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _registration is None:
            return
        pid, module, matches, create = _owner
        try:
            factory.require_model_runner_factory(_registration)
            if (
                pid != os.getpid()
                or importlib.import_module(module.__name__) is not module
                or module.matches is not matches
                or module.create is not create
                or _registration.matches is not matches
                or _registration.create is not create
            ):
                _fatal("HY4 PCP runner registration changed owner")
            module.check_health()
        except ExtensionCommitError:
            raise
        except Exception as error:
            _failure = _failure or f"HY4 PCP runner publication is unhealthy: {error}"
            raise ExtensionCommitError(_failure) from error


def require_active():
    check_health()
    if _registration is None:
        raise ValueError("HY4 PD requires explicit hy4-pcp-dfly capability")


def activate(context):
    global _registration, _owner, _failure
    selected = (
        "hy4-pcp-dfly" in context.requested_capabilities
        or OWNER in context.requested_extensions
    )
    with _lock:
        check_health()
        if not selected:
            if _registration is not None:
                _fatal("HY4 PCP runner publication cannot be disabled")
            return
        version = Version(context.vllm_version or "0")
        if version.epoch or version.release[:3] != (0, 33, 0):
            raise ValueError("HY4 PCP DFly requires Core 0.33.0 interfaces")
        if _registration is not None:
            return
        from hcu_adapter.features.hy4 import pcp_profile

        from .models.hy4_standard import pcp_runner

        if pcp_runner._profile_selected():
            pcp_profile.install()
        pcp_runner.prepare()
        try:
            _registration = factory.register_model_runner_factory(
                OWNER, matches=pcp_runner.matches, create=pcp_runner.create
            )
            _owner = (os.getpid(), pcp_runner, pcp_runner.matches, pcp_runner.create)
            check_health()
        except Exception as error:
            _failure = _failure or f"HY4 PCP runner publication failed; restart: {error}"
            raise ExtensionCommitError(_failure) from error


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-hy4-pcp-dfly",
        api_version=1,
        priority=104,
        capabilities=frozenset({"hy4-pcp-dfly"}),
        activate=activate,
        check_health=check_health,
    )
