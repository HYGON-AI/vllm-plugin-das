# SPDX-License-Identifier: Apache-2.0
"""Framework-free discovery of the owned HCU V41 PCP/host target model."""
from __future__ import annotations

import os
import sys
from threading import RLock
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from .dsv41_activation import _require_release
from .environment import _flag
from hcu_adapter.fp4_indexer import selected as fp4_selected

_FLAG = 'VLLM_HCU_DSV41_PCP_ATTENTION'
_HOST_FLAG = 'VLLM_HCU_DSV41_ENGRAM_HOST'
_FACTOR = 'VLLM_HCU_DAS_V41_PCP_ATTENTION'
_ARCH = 'DeepseekV41ForCausalLM'
_NATIVE = ('vllm.models.deepseek_v41', 'DeepseekV41ForCausalLM')
_PLUGIN = ('hcu_vllm_das.models.v41_pcp', 'HcuV41PcpForCausalLM')
_state = None
_failure = None
_publishing = False
_lock = RLock()


def _fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = 'fork inherited incomplete V41 PCP publication'


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_fork)


def _is_hcu():
    from vllm.platforms import current_platform
    return current_platform.is_rocm() and current_platform.device_name == 'hcu'


def _key(entry):
    return getattr(entry, 'module_name', None), getattr(entry, 'class_name', None)


def _function(fn):
    return getattr(fn, '__func__', fn)


def pcp_enabled():
    return _state['enabled'] if _state is not None else _flag(_FLAG, default=False)


def host_enabled():
    return _state['host'] if _state is not None else _flag(_HOST_FLAG, default=False)


def policy_factor():
    return (pcp_enabled(), host_enabled(), True) if fp4_selected() else (pcp_enabled(), host_enabled())


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _adapter_bindings_valid():
    """Read only: the adapter still owns the scorer and validator we injected.

    The adapter refuses a second, different owner, so a mismatch here means some
    other component rebound the classes mid-flight. Never rebinds or repairs.
    """
    from hcu_adapter.kernels import pcp_fork

    if _state is None:
        return True
    if not (_state['enabled'] or _state['host'] or _state.get('fp4', False)):
        return True
    if pcp_fork.bound_reference() is None:
        return False
    if (_state['enabled'] or _state['host']) and pcp_fork.bound_engram_validator() is None:
        return False
    return True


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _state is None:
            if not _is_hcu():
                return
            raise RuntimeError('HCU V41 PCP extension is not active')
        try:
            from vllm import envs
            from vllm.model_executor.models import ModelRegistry
            if (ModelRegistry is not _state['registry']
                    or ModelRegistry.models.get(_ARCH) is not _state['entry']
                    or _function(ModelRegistry.register_model) is not _state['register']
                    or _flag(_FLAG, default=False) != _state['enabled']
                    or _flag(_HOST_FLAG, default=False) != _state['host']
                    or fp4_selected() != _state.get('fp4', False)
                    or envs.environment_variables.get(_FACTOR) is not policy_factor
                    or not _adapter_bindings_valid()):
                _fatal('V41 PCP ownership/policy changed; restart process')
            loaded = sys.modules.get('hcu_vllm_das.models.v41_pcp')
            if loaded is not None:
                loaded.check_native_contract()
        except ExtensionCommitError:
            raise
        except BaseException as error:
            try:
                _fatal('V41 PCP health contract is unavailable; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error


def activate_extension(context):
    global _state, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal('recursive V41 PCP activation')
        _require_release(context.vllm_version)
        if _state is not None:
            check_health()
            return
        if not _is_hcu():
            return
        pcp_selected = _flag(_FLAG, default=False)
        host_selected = _flag(_HOST_FLAG, default=False)
        fp4 = fp4_selected()
        selected = pcp_selected or host_selected or fp4
        _publishing = True
        try:
            import inspect
            from vllm import envs
            from vllm.model_executor.models import ModelRegistry
            entry = ModelRegistry.models.get(_ARCH)
            expected = (_NATIVE, _PLUGIN) if selected else (_NATIVE,)
            if _key(entry) not in expected:
                raise RuntimeError('V41 PCP target model has a foreign or missing owner')
            if tuple(inspect.signature(ModelRegistry.register_model).parameters) != ('model_arch', 'model_cls'):
                raise RuntimeError('V41 public ModelRegistry registration signature changed')
            if envs.environment_variables.get(_FACTOR) not in (None, policy_factor):
                raise RuntimeError('V41 PCP compile factor has a foreign owner')
            register = _function(ModelRegistry.register_model)
            # The PCP classes now live in the adapter, which may not import the
            # plugin. Inject the plugin-owned FP8 scorer and Engram validator
            # before any owned model can be constructed.
            if selected:
                from .v41_pcp import bind_adapter_classes
                bind_adapter_classes()
            if selected and _key(entry) == _NATIVE:
                ModelRegistry.register_model(_ARCH, ':'.join(_PLUGIN))
            entry = ModelRegistry.models.get(_ARCH)
            if _key(entry) != (_PLUGIN if selected else _NATIVE):
                raise RuntimeError('V41 PCP model publication lost its owner')
            envs.environment_variables[_FACTOR] = policy_factor
            _state = dict(enabled=pcp_selected, host=host_selected, fp4=fp4, registry=ModelRegistry, entry=entry, register=register)
            check_health()
        except BaseException as error:
            try:
                _fatal('V41 PCP publication failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(name='das-hcu-v41-pcp', api_version=1, priority=116,
        capabilities=frozenset({'hcu-v41-pcp-eager-attention', 'hcu-v41-private-engram-host'}),
        activate=activate_extension, check_health=check_health)
