# SPDX-License-Identifier: Apache-2.0
"""Explicit independent publication of a distinct checkpoint format."""

from __future__ import annotations

from importlib import import_module
from contextlib import contextmanager
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .extension import _require_vllm_033_release_line

CAPABILITY = "hcu-slimquant-w4a8"
METHOD = "slimquant_w4a8"
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
        raise ExtensionCommitError("SlimQuant activation failed; restart the process")
    if _activating:
        raise RuntimeError("SlimQuant activation re-entered before completion")
    _activating = True
    try:
        yield
    finally:
        _activating = False


def activate(context):
    global _published, _failure
    _require_vllm_033_release_line(context.vllm_version)
    if CAPABILITY not in context.requested_capabilities:
        raise RuntimeError("SlimQuant requires explicit hcu-slimquant-w4a8 capability")
    with _lock, _attempt():
        if _failure:
            raise ExtensionCommitError(
                "SlimQuant publication failed; restart the process"
            )
        if _published is not None:
            check_health()
            return
        # Importing this implementation validates the public PLE extension API
        # before the public registry is changed. No official method is replaced.
        ple = import_module("vllm.models.qwen4_exp.common.ngram_embedding")
        if not callable(getattr(ple, "Qwen4ExpPLEPerRowEmbeddingMethod", None)):
            raise RuntimeError("SlimQuant requires the public per-row PLE Core API")
        config = import_module("hcu_vllm_das.slimquant").SlimQuantConfig
        registry = import_module("vllm.model_executor.layers.quantization")
        try:
            existing = registry.get_quantization_config(METHOD)
        except ValueError:
            existing = None
        if existing is not None and existing is not config:
            raise RuntimeError(
                "SlimQuant quantization method already has another owner"
            )
        try:
            registry.register_quantization_config(METHOD)(config)
            _published = config
            check_health()
        except BaseException as error:
            _failure = True
            raise ExtensionCommitError(
                "SlimQuant publication failed; restart the process"
            ) from error


def check_health():
    if _failure or _published is None:
        raise RuntimeError("SlimQuant has no healthy publication")
    registry = import_module("vllm.model_executor.layers.quantization")
    try:
        owner = registry.get_quantization_config(METHOD)
    except BaseException as error:
        raise ExtensionCommitError(
            "SlimQuant registry unavailable; restart the process"
        ) from error
    if owner is not _published:
        raise ExtensionCommitError(
            "SlimQuant registry owner changed; restart the process"
        )


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-slimquant",
        api_version=1,
        priority=70,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
