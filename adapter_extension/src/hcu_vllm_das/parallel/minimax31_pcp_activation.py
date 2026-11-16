# SPDX-License-Identifier: Apache-2.0
"""Explicit PCP/EP capability, composed with MiniMax SlimQuant."""
import os
from threading import RLock

from packaging.version import Version
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

ENTRY = "minimax31_pcp_ep"
CAPABILITY = "minimax31-pcp-ep"
_lock = RLock()
_publication = None
_failure = None
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
    _failure = _failure or reason
    if error is not None:
        raise ExtensionCommitError(_failure) from error
    raise ExtensionCommitError(_failure)


def check_health():
    _check_process()
    if _failure is not None:
        _fatal(_failure)
    if _publication != os.getpid():
        _fatal("MiniMax PCP/EP is not published in this process")
    try:
        from .minimax31_pcp import check_health as model_health
        from .minimax31_ipc import check_pcp_ep_ipc_health
        model_health()
        check_pcp_ep_ipc_health()
    except BaseException as error:
        _fatal("MiniMax PCP/EP ownership or compatibility changed", error)


def activate(context):
    global _publication
    _check_process()
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publication is not None:
            check_health()
            return
        if ENTRY not in context.requested_extensions and CAPABILITY not in context.requested_capabilities:
            raise RuntimeError("MiniMax PCP/EP requires an explicit extension request")
        if Version(context.vllm_version or "0").release[:2] != (0, 33):
            raise RuntimeError("MiniMax PCP/EP requires the supported vLLM 0.33 build")
        from ..quantization.minimax31_activation import check_health as quant_health
        quant_health()
        from hcu_adapter.bridges.vllm_model_load import register_post_model_load
        from .minimax31_pcp import install_minimax31_pcp
        from .minimax31_ipc import install_pcp_ep_ipc_loader
        from .minimax31_ep_state import after_minimax31_ep_load
        try:
            install_minimax31_pcp()
            install_pcp_ep_ipc_loader()
            register_post_model_load("minimax31-pcp-ep.expert-map", after_minimax31_ep_load)
            _publication = os.getpid()
            check_health()
        except BaseException as error:
            _fatal("MiniMax PCP/EP activation failed after publication began", error)


def extension_spec():
    return ExtensionSpec(
        name=ENTRY, api_version=1, priority=50,
        capabilities=frozenset({CAPABILITY}), activate=activate,
        check_health=check_health)
