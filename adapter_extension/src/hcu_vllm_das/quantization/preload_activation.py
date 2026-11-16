# SPDX-License-Identifier: Apache-2.0
"""Explicit, experimental, process-local K3.1 SlimQuant preload publication.

Not present in the production wheel entry points. A task-scoped diagnostic
entry point may request this capability; it does not enable forward.
"""
from __future__ import annotations
import os
from threading import RLock
from packaging.version import Version
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

ENTRY = "kimi31_slimquant_exp"
CAPABILITY = "kimi31-slimquant-preload-experimental"
_lock=RLock()
_publication=None
_failure=None
_pid = os.getpid()


def _check_process():
    global _pid, _lock
    pid = os.getpid()
    if pid != _pid:
        # Check before entering locks inherited from arbitrary parent threads.
        _lock = RLock()
        if _publication is not None or _failure is not None:
            _fatal("extension process state inherited across fork; use spawn")
        _pid = pid



def _fatal(reason, error=None):
    global _failure
    if _failure is None: _failure=reason
    if error is None: raise ExtensionCommitError(_failure)
    raise ExtensionCommitError(_failure) from error


def _owned(pub):
    pid, q, registry, config, outer, inner, outer_name, inner_name, outer_record, inner_record = pub
    return (pid == os.getpid() and q._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get("slimquant_w4a8") is config
            and "slimquant_w4a8" in q.QUANTIZATION_METHODS
            and registry.models.get(outer_name) is outer_record
            and registry.models.get(inner_name) is inner_record
            and outer_record.model_cls is outer and inner_record.model_cls is inner)


def check_health():
    _check_process()
    with _lock:
        if _failure is not None: _fatal(_failure)
        if _publication is None: raise RuntimeError("K3.1 SlimQuant candidate was not activated")
        if not _owned(_publication): _fatal("K3.1 SlimQuant registry ownership or process identity changed")


def activate(context):
    global _publication
    _check_process()
    with _lock:
        if _failure is not None: _fatal(_failure)
        if _publication is not None:
            if not _owned(_publication): _fatal("K3.1 SlimQuant registry state changed after publication")
            return
        if ENTRY not in context.requested_extensions and CAPABILITY not in context.requested_capabilities:
            raise RuntimeError("experimental K3.1 preload requires explicit extension request")
        v=Version(context.vllm_version or "0")
        if v.epoch != 0 or v.release[:2] != (0,33):
            raise RuntimeError("K3.1 SlimQuant candidate requires vLLM 0.33")
        try:
            from hcu_adapter.bridges.vllm_kimi31_preload import selected_mode, require_healthy
        except ModuleNotFoundError as error:
            if error.name != "hcu_adapter.bridges.vllm_kimi31_preload":
                raise
            raise RuntimeError(
                "K3.1 preload requires the optional Kimi preload bridge in the companion adapter"
            ) from error
        if selected_mode() != "nvidia-load-only":
            raise RuntimeError("K3.1 SlimQuant candidate requires nvidia-load-only")
        require_healthy()
        from vllm.model_executor.layers import quantization as q
        from vllm.model_executor.models.registry import ModelRegistry as registry
        from .slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig as config
        from .kimi31_load_model import (Kimi31SlimQuantLoadOnly as outer,
            Kimi31SlimQuantLoadOnlyInner as inner, OUTER_ARCH as outer_name, INNER_ARCH as inner_name)
        if "slimquant_w4a8" in q.QUANTIZATION_METHODS or "slimquant_w4a8" in q._CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
            raise RuntimeError("foreign SlimQuant quantization owner already registered")
        if outer_name in registry.models or inner_name in registry.models:
            raise RuntimeError("foreign K3.1 candidate model already registered")
        try:
            # Registry publication is non-transactional; any failure after the
            # first write is commit-fatal and never advertised as a rollback.
            q.register_quantization_config("slimquant_w4a8")(config)
            registry.register_model(inner_name,inner)
            registry.register_model(outer_name,outer)
            _publication=(os.getpid(),q,registry,config,outer,inner,outer_name,inner_name,
                          registry.models[outer_name],registry.models[inner_name])
            if not _owned(_publication): _fatal("K3.1 candidate publication verification failed")
        except BaseException as error:
            _fatal("K3.1 candidate registry publication failed",error)


def extension_spec():
    return ExtensionSpec(name=ENTRY, api_version=1, priority=60,
                         capabilities=frozenset({CAPABILITY}), activate=activate,
                         check_health=check_health)
