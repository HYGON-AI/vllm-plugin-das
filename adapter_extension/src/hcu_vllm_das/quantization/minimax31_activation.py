# SPDX-License-Identifier: Apache-2.0
"""Explicit MiniMax SlimQuant registration through the adapter extension ABI."""
from __future__ import annotations

import os
from threading import RLock
from packaging.version import Version
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

ENTRY = "minimax31_slimquant"
CAPABILITY = "minimax31-slimquant"
_lock = RLock()
_publication = None
_failure = None


def _fatal(reason, error=None):
    global _failure
    if _failure is None:
        _failure = reason
    if error is None:
        raise ExtensionCommitError(_failure)
    raise ExtensionCommitError(_failure) from error


def _owned(pub):
    pid, registry, config = pub
    return (pid == os.getpid()
            and registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get("slimquant_w4a8") is config
            and registry.QUANTIZATION_METHODS.count("slimquant_w4a8") == 1)


def check_health():
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publication is None:
            raise RuntimeError("MiniMax SlimQuant extension has not been activated")
        if not _owned(_publication):
            _fatal("MiniMax SlimQuant registry ownership/process changed")


def activate(context):
    global _publication
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publication is not None:
            if not _owned(_publication):
                _fatal("MiniMax SlimQuant registry changed after publication")
            return
        if ENTRY not in context.requested_extensions and CAPABILITY not in context.requested_capabilities:
            raise RuntimeError("MiniMax SlimQuant requires an explicit extension request")
        version = Version(context.vllm_version or "0")
        if version.epoch != 0 or version.release[:2] != (0, 33):
            raise RuntimeError("MiniMax SlimQuant requires vLLM 0.33")
        from vllm.model_executor.layers import quantization as registry
        from .minimax31 import MiniMax31SlimQuantConfig
        if ("slimquant_w4a8" in registry.QUANTIZATION_METHODS
                or "slimquant_w4a8" in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG):
            raise RuntimeError("foreign SlimQuant quantization owner already registered")
        from hcu_adapter.bridges.vllm_model_load import register_post_model_load
        from .minimax31 import after_minimax31_model_load
        try:
            register_post_model_load("minimax31-slimquant.router-state", after_minimax31_model_load)
            registry.register_quantization_config("slimquant_w4a8")(MiniMax31SlimQuantConfig)
            _publication = (os.getpid(), registry, MiniMax31SlimQuantConfig)
            if not _owned(_publication):
                _fatal("MiniMax SlimQuant registration verification failed")
        except BaseException as error:
            _fatal("MiniMax SlimQuant registration failed after publication began", error)


def extension_spec():
    return ExtensionSpec(name=ENTRY, api_version=1, priority=60,
                         capabilities=frozenset({CAPABILITY}), activate=activate,
                         check_health=check_health)
