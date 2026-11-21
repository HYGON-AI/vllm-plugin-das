# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""ByteLevel provider; Adapter owns admission, synchronization and fatal health."""
from importlib import import_module
import os

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

_FLAG = "VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL"
_state = None


def _mode():
    value = os.environ.get(_FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_FLAG} must be 0 or 1")
    return value == "1"


def _owns_publication():
    _, mode, module, cls, constructor = _state
    return not mode or (
        import_module("vllm.v1.engine.detokenizer") is module
        and module.FastIncrementalDetokenizer is cls
        and cls.__init__ is constructor
    )


def check_health():
    """Read-only provider check; Adapter latches any published failure."""
    if _state is None:
        raise ValueError("ByteLevel extension has not activated")
    try:
        valid = _mode() == _state[1] and _owns_publication()
    except BaseException as error:
        raise ExtensionCommitError("ByteLevel published policy/owner unavailable") from error
    if not valid:
        raise ExtensionCommitError("ByteLevel published policy/owner changed; restart")
    if _state[0] != os.getpid():
        raise ValueError("ByteLevel extension requires activation after fork")


def activate(context):
    """Called after the Adapter's version/platform/runtime admission."""
    global _state
    if _state is not None:
        try:
            valid = _mode() == _state[1] and _owns_publication()
        except BaseException as error:
            raise ExtensionCommitError("ByteLevel published state unavailable") from error
        if not valid:
            raise ExtensionCommitError("ByteLevel published policy/owner changed; restart")
        _state = (os.getpid(), *_state[1:])
        return
    mode = _mode()
    required = (
        'das_hcu_hy4_frontend' in context.requested_extensions
        or 'bytelevel-prompt-tail' in context.requested_capabilities
    )
    if required and not mode:
        raise ValueError(f'ByteLevel provider requires {_FLAG}=1')
    if not mode:
        _state = (os.getpid(), False, None, None, None)
        return
    from .hy4_detokenizer import install
    module = import_module("vllm.v1.engine.detokenizer")
    cls = module.FastIncrementalDetokenizer
    original = cls.__init__
    try:
        install(module)
        _state = (os.getpid(), True, module, cls, cls.__init__)
    except BaseException as error:
        if cls.__init__ is not original:
            raise ExtensionCommitError("ByteLevel publication failed; restart") from error
        raise


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-hy4-frontend", api_version=1, priority=80,
        capabilities=frozenset({"bytelevel-prompt-tail"}),
        activate=activate, check_health=check_health)
