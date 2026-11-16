# SPDX-License-Identifier: Apache-2.0
"""Publish the opt-in AutoAWQ provider using the official config registry."""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib import import_module
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .environment import lightop_awq_enabled
from .extension import _require_vllm_033_release_line

CAPABILITY = "lightop-autoawq"
METHODS = ("awq", "awq_marlin", "auto_awq")
FACTOR = "VLLM_HCU_DAS_LIGHTOP_AWQ_ENABLED"
_lock = RLock()
_published = None
_failure = False
_activating = False


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _activating:
        _failure = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


@contextmanager
def _attempt():
    global _activating
    if _failure:
        raise ExtensionCommitError("AutoAWQ activation failed; restart the process")
    if _activating:
        raise RuntimeError("AutoAWQ activation re-entered before completion")
    _activating = True
    try:
        yield
    finally:
        _activating = False


def activate(context):
    global _published, _failure
    _require_vllm_033_release_line(context.vllm_version)
    if CAPABILITY not in context.requested_capabilities:
        raise RuntimeError("AutoAWQ requires explicit lightop-autoawq capability")
    with _lock, _attempt():
        if _published is not None:
            check_health()
            return
        registry = import_module("vllm.model_executor.layers.quantization")
        original = import_module(
            "vllm.model_executor.layers.quantization.auto_awq"
        ).AutoAWQConfig
        config = import_module("hcu_vllm_das.autoawq").HcuAutoAWQConfig
        envs = import_module("vllm.envs")
        for method in METHODS:
            if registry.get_quantization_config(method) is not original:
                raise RuntimeError(
                    "AutoAWQ registry already has another owner: " + method
                )
        if not callable(getattr(envs, "compile_factors", None)):
            raise RuntimeError("Official compilation environment registry unavailable")
        existing = envs.environment_variables.get(FACTOR)
        if existing is not None and existing is not lightop_awq_enabled:
            raise RuntimeError("AutoAWQ compile policy already has another owner")
        # This irreversible publication includes the custom-op/FakeTensor
        # registration. A partial failure poisons the extension, with no retry.
        try:
            import_module("hcu_vllm_das.autoawq_ops")
            envs.environment_variables[FACTOR] = lightop_awq_enabled
            for method in METHODS:
                registry.register_quantization_config(method)(config)
            _published = config
            check_health()
        except BaseException as error:
            _failure = True
            raise ExtensionCommitError(
                "AutoAWQ publication failed; restart the process"
            ) from error


def check_health():
    if _failure or _published is None:
        raise ExtensionCommitError("AutoAWQ has no healthy publication")
    registry = import_module("vllm.model_executor.layers.quantization")
    envs = import_module("vllm.envs")
    if any(
        registry.get_quantization_config(name) is not _published for name in METHODS
    ):
        raise ExtensionCommitError(
            "AutoAWQ registry owner changed; restart the process"
        )
    if envs.environment_variables.get(FACTOR) is not lightop_awq_enabled:
        raise ExtensionCommitError(
            "AutoAWQ compile policy owner changed; restart the process"
        )


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-autoawq",
        api_version=1,
        priority=75,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
