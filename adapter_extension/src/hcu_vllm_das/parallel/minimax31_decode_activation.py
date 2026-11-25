# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in for MiniMax DP8 decode and strict dual-role preload."""
import os
from threading import RLock
from packaging.version import Version
from hcu_adapter.extensions import ExtensionSpec,ExtensionCommitError

ENTRY="minimax31_decode"
_lock=RLock()
_pid=os.getpid()
_published=False
_failed=False
_draft_record=None


def check_health():
    if os.getpid()!=_pid or _failed or not _published:
        raise ExtensionCommitError("MiniMax decode owner is not healthy in this process")
    from .minimax31_decode import check_health as model_health
    model_health()
    from .minimax31_dspark_admission import check_health as config_health
    config_health()
    from .minimax31_decode_ipc import LOAD_FORMAT,DecodeIpcLoader
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER
    if _LOAD_FORMAT_TO_MODEL_LOADER.get(LOAD_FORMAT) is not DecodeIpcLoader:
        raise ExtensionCommitError("MiniMax decode loader ownership drift")
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.model_executor.layers import quantization
    from ..quantization.minimax31_draft import MiniMax31DraftInt8Config
    if (ModelRegistry.models.get('DSparkMiniMaxDraftModel') is not _draft_record or
            quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get('minimax31_dspark_int8') is not MiniMax31DraftInt8Config):
        raise ExtensionCommitError('MiniMax draft model/quantization ownership drift')


def activate(context):
    global _published,_failed,_draft_record
    if os.getpid()!=_pid:
        raise ExtensionCommitError("MiniMax decode owner inherited across fork; use spawn")
    with _lock:
        if _failed:
            raise ExtensionCommitError("MiniMax decode activation previously failed")
        if _published:
            check_health();return
        if ENTRY not in context.requested_extensions and "minimax31-decode" not in context.requested_capabilities:
            raise RuntimeError("MiniMax decode requires explicit selection")
        if Version(context.vllm_version or "0").release[:2]!=(0,33):
            raise RuntimeError("MiniMax decode requires vLLM 0.33")
        from ..quantization.minimax31_activation import check_health as quant_health
        quant_health()
        from hcu_adapter.bridges.vllm_model_load import register_post_model_load
        from .minimax31_decode_ipc import register_loader
        from .minimax31_markov import after_minimax31_markov_load
        from .minimax31_decode import install
        from .minimax31_dspark_admission import install as install_draft_config
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.model_executor.layers import quantization
        from ..quantization.minimax31_draft import MiniMax31DraftInt8Config
        if ('DSparkMiniMaxDraftModel' in ModelRegistry.models or
                'minimax31_dspark_int8' in quantization.QUANTIZATION_METHODS):
            raise RuntimeError('foreign MiniMax DSpark model/quantization registration')
        try:
            quantization.register_quantization_config('minimax31_dspark_int8')(MiniMax31DraftInt8Config)
            ModelRegistry.register_model('DSparkMiniMaxDraftModel',
                'hcu_vllm_das.models.minimax31_dspark:DSparkMiniMaxDraftModel')
            _draft_record=ModelRegistry.models['DSparkMiniMaxDraftModel']
            install_draft_config()
            register_loader()
            install()
            register_post_model_load("minimax31-decode.markov-greedy", after_minimax31_markov_load)
            _published=True
            check_health()
        except BaseException as exc:
            _failed=True
            raise ExtensionCommitError("MiniMax decode failed after publication began") from exc


def extension_spec():
    return ExtensionSpec(name=ENTRY,api_version=1,priority=50,
        capabilities=frozenset({"minimax31-decode"}),activate=activate,check_health=check_health)
