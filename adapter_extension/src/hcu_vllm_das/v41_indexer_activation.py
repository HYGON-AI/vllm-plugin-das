# SPDX-License-Identifier: Apache-2.0
"""Framework-free discovery and public OOT publication of an opt-in indexer."""
from __future__ import annotations

import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .dsv41_activation import _require_release
from .environment import _flag
from hcu_adapter.fp4_indexer import selected as fp4_selected

_state = None
_failure = None
_publishing = False
_lock = RLock()
_FLAG = 'VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE'
_FACTOR = 'VLLM_HCU_DAS_V41_INDEXER_REFERENCE'


def _fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = 'fork inherited incomplete V41 indexer publication'


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_fork)


def _is_hcu():
    from vllm.platforms import current_platform
    return current_platform.is_rocm() and current_platform.device_name == 'hcu'


def policy_factor():
    if _state is not None and _state.get('fp4', False):
        return ('fp4-lightop', _state['digest'])
    return _state['enabled'] if _state is not None else _flag(_FLAG, default=True)


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _state is None:
            if not _is_hcu():
                return
            raise RuntimeError('V41 reference indexer is not active')
        from vllm import envs
        from vllm.model_executor import custom_op
        try:
            valid = (
                _flag(_FLAG, default=True) == _state['enabled']
                and fp4_selected() == _state.get('fp4', False)
                and custom_op.op_registry_oot is _state['registry']
                and custom_op.op_registry_oot.get('SparseAttnIndexer') is _state['class']
                and envs.environment_variables.get(_FACTOR) is policy_factor
                and _state['native'].forward_hip is _state['forward']
                and _state['class'].name == _state['native'].name
            )
            if not valid:
                _fatal('V41 indexer ownership/policy changed; restart process')
            if _state.get('fp4', False):
                from . import fp4_packer
                from .fp4_indexer_lightop import load_providers
                from .fp4_indexer import check_native_contract
                fp4_packer.check_health()
                load_providers()
                check_native_contract()
        except ExtensionCommitError:
            raise
        except BaseException as error:
            try:
                _fatal('V41 indexer public health is unavailable')
            except ExtensionCommitError as refusal:
                raise refusal from error


def activate_extension(context):
    global _state, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal('recursive V41 indexer activation')
        if _state is not None:
            try:
                _require_release(context.vllm_version)
                check_health()
            except BaseException as error:
                try:
                    _fatal('V41 indexer contract changed after publication')
                except ExtensionCommitError as refusal:
                    raise refusal from error
            return
        _require_release(context.vllm_version)
        if not _is_hcu():
            return
        selected = _flag(_FLAG, default=True)
        fp4 = fp4_selected()
        digest = None
        if fp4:
            from . import fp4_packer
            from .fp4_indexer_lightop import load_providers
            from hcu_adapter import hcu_env
            if hcu_env.resolve('HCU_VLLM_SPARSE_INDEXER_IMPL', host='vllm')[0] != 'stock':
                raise RuntimeError('C4 and V41 FP4 cannot own the same OOT key')
            fp4_packer.check_health()
            if fp4_packer._state is None or fp4_packer._state['mode'] != 'triton':
                raise RuntimeError('FP4 requires published triton lowering')
            load_providers()
            import hashlib
            from pathlib import Path
            from hcu_adapter import v41_fp8_indexer
            digest = hashlib.sha256()
            # The FP4 branch executes through the adapter-owned indexer base, so
            # its source is part of the pinned contract, not just the plugin's.
            for path in (Path(__file__).with_name('fp4_indexer.py'),
                         Path(__file__).with_name('fp4_indexer_lightop.py'),
                         Path(__file__).with_name('v41_indexer.py'),
                         Path(v41_fp8_indexer.__file__)):
                digest.update(path.read_bytes())
            digest = digest.hexdigest()
        _publishing = True
        try:
            from vllm import envs
            from vllm.model_executor import custom_op
            from .v41_indexer import HcuV41SparseAttnIndexer, SparseAttnIndexer, register_op
        except BaseException as error:
            try:
                _fatal('V41 indexer native preparation failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False
        registry = custom_op.op_registry_oot
        if registry.get('SparseAttnIndexer') not in (None, HcuV41SparseAttnIndexer):
            raise RuntimeError('V41 reference indexer has a foreign OOT owner')
        if envs.environment_variables.get(_FACTOR) not in (None, policy_factor):
            raise RuntimeError('V41 reference indexer compile factor has a foreign owner')
        _publishing = True
        try:
            register_op()
            envs.environment_variables[_FACTOR] = policy_factor
            if 'SparseAttnIndexer' not in registry:
                SparseAttnIndexer.register_oot()(HcuV41SparseAttnIndexer)
            # The public OOT key is the class name; the compilation switch is
            # the original lower-case op name. Preserve the latter for users.
            HcuV41SparseAttnIndexer.name = SparseAttnIndexer.name
            _state = dict(enabled=selected, fp4=fp4, digest=digest, registry=registry, **{
                'class': HcuV41SparseAttnIndexer, 'native': SparseAttnIndexer,
                'forward': SparseAttnIndexer.forward_hip})
            if fp4:
                from hcu_adapter import registry as adapter_registry
                adapter_registry.register(
                    op='vllm_fp4_indexer', impl='lightop',
                    target='vllm.model_executor.layers.sparse_attn_indexer.SparseAttnIndexer',
                    mechanism=adapter_registry.MECH_SEAM, source=__name__,
                    detail='FP4 branch of existing V41 OOT; native/model gates pending')
            check_health()
        except BaseException as error:
            try:
                _fatal('V41 indexer publication failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(name='das-hcu-v41-indexer', api_version=1, priority=115,
        capabilities=frozenset({'hcu-v41-fp8-indexer-eager-reference',
                                'hcu-v41-fp8-indexer-graph-break'}),
        activate=activate_extension, check_health=check_health)
