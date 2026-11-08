# SPDX-License-Identifier: Apache-2.0
"""Explicit independent publication of a distinct checkpoint format."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from contextlib import contextmanager
import inspect
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
_packed_provider = None
_packed_provider_pid = None


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _activating or _packed_provider is not None:
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


def require_usable():
    """Reject poisoned execution without requiring activation for pure metadata."""
    if _failure:
        raise ExtensionCommitError("SlimQuant owner failed; restart the process")


def prepare_packed_provider():
    """Prepare the optional vendor only for explicit packed selection.

    Host health is queried outside the owner lock. A clean absent top-level
    dependency does not poison expanded INT8; partial native import does.
    """
    global _packed_provider, _packed_provider_pid, _failure
    from hcu_adapter.bridges.vllm import require_extension_healthy

    require_extension_healthy(CAPABILITY)
    try:
        with _lock, _attempt():
            if _packed_provider is not None:
                if _packed_provider_pid != os.getpid():
                    raise ExtensionCommitError(
                        "SlimQuant native provider cannot be inherited; use a fresh spawned process"
                    )
                check_health()
                return _packed_provider
            try:
                dependency = find_spec("boltops")
            except BaseException as error:
                _failure = True
                raise ExtensionCommitError(
                    "SlimQuant native dependency discovery failed; restart the process"
                ) from error
            if dependency is None:
                raise RuntimeError("packed SlimQuant requires optional BoltOps")
            try:
                module = import_module("boltops.fused_moe")
                function = module.fused_experts
                normalizer = module.normalize_moe_activation
                for candidate in (function, normalizer):
                    implementations = ((candidate,) if inspect.isroutine(candidate)
                                       else (candidate, candidate.__call__))
                    if (not callable(candidate)
                            or any(check(implementation)
                                   for implementation in implementations
                                   for check in (inspect.iscoroutinefunction,
                                                 inspect.isasyncgenfunction,
                                                 inspect.isgeneratorfunction))):
                        raise TypeError("BoltOps entry must be synchronous")
                # Public normalization loads the shipped lazy backend before
                # apply, keeping partial imports inside this guarded attempt.
                if normalizer("silu") != ("silu", True):
                    raise ValueError("BoltOps gated SiLU contract unavailable")
                inspect.signature(function).bind(
                    object(), object(), object(), object(), object(),
                    inplace=False, activation="silu", use_int4_w4a8=True,
                    per_channel_quant=True, global_num_experts=2,
                    w1_scale=object(), w2_scale=object(), output_dtype=object(),
                )
                check_health()
            except BaseException as error:
                # Latch before any exception formatting, including bad __str__.
                _failure = True
                raise ExtensionCommitError(
                    "SlimQuant native preparation failed; restart the process"
                ) from error
            _packed_provider = function
            _packed_provider_pid = os.getpid()
            return function
    except BaseException:
        if _failure:
            # Publish the failed owner through the public host health API only
            # after releasing the owner lock. Never clear or repair reports.
            try:
                require_extension_healthy(CAPABILITY)
            except BaseException:
                pass
        raise


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-slimquant",
        api_version=1,
        priority=70,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
