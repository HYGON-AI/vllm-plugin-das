# SPDX-License-Identifier: Apache-2.0
"""Framework-free discovery of the owned HCU V41 PCP/host target model."""
from __future__ import annotations

import logging
import os
import sys
from threading import RLock
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from .adapter_api import adapter_module as _adapter_module
from .dsv41_activation import _require_release
from .environment import _flag
fp4_selected = _adapter_module('fp4_indexer').selected
from .inverse_rope_policy import selected_mode as inverse_rope_mode

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
_logger = logging.getLogger(f'vllm.{__name__}')


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


def _profile_policy():
    from .environment import dsv41_profile_worker_enabled

    return dsv41_profile_worker_enabled()


def _install_profile_worker():
    from vllm.platforms import current_platform
    from .dsv41.profile_worker import install
    from hcu_adapter.bridges.vllm_dsv41 import register_compatibility

    install(current_platform)
    register_compatibility()


def _q_rope_policy():
    from hcu_adapter.kernels import pcp_fork

    selected = getattr(pcp_fork, 'q_rope_selected', None)
    return selected is not None and selected()


def policy_factor():
    policy = (pcp_enabled(), host_enabled(), True) if fp4_selected() else (pcp_enabled(), host_enabled())
    if pcp_enabled():
        policy = (*policy, _profile_policy())
        mode = inverse_rope_mode()
        if mode == 'hcu':
            policy = (*policy, mode)
        if _q_rope_policy():
            policy = (*policy, 'q_rope_hcu')
    return policy


def _cache_builder(config):
    from .dsa_cache_layout import builder_for_config
    return builder_for_config(config)


def _localize_cache(config, cache_config):
    from .dsa_cache_layout import localize_kv_cache
    return localize_kv_cache(config, cache_config)


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
    check_q_rope = getattr(pcp_fork, 'require_q_rope_healthy', None)
    if check_q_rope is not None:
        try:
            check_q_rope()
        except pcp_fork.PcpAttentionError:
            return False
    if pcp_fork.bound_reference() is None:
        return False
    if (_state['enabled'] or _state['host']) and pcp_fork.bound_engram_validator() is None:
        return False
    if _state.get('inverse_rope_mode', 'stock') == 'hcu':
        from .inverse_rope_pcp import prepare_forward_mqa
        if pcp_fork.bound_forward_mqa_provider() is not prepare_forward_mqa:
            return False
    if _state.get('layer_split_layout') is not None:
        from .v41_pcp import _validate_transfer
        return pcp_fork.bound_kv_transfer_validator() is _validate_transfer
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
            layout = _state.get('layer_split_layout')
            if layout is not None:
                layout.require_kv_cache_layout('das-hcu-v41-pcp', _cache_builder, _localize_cache)
            if (ModelRegistry is not _state['registry']
                    or ModelRegistry.models.get(_ARCH) is not _state['entry']
                    or _function(ModelRegistry.register_model) is not _state['register']
                    or _flag(_FLAG, default=False) != _state['enabled']
                    or _flag(_HOST_FLAG, default=False) != _state['host']
                    or (_state['enabled'] and _profile_policy() != _state['profile_worker'])
                    or fp4_selected() != _state.get('fp4', False)
                    or inverse_rope_mode() != _state.get('inverse_rope_mode', 'stock')
                    or envs.environment_variables.get(_FACTOR) is not policy_factor
                    or not _adapter_bindings_valid()
                    or (_state['enabled'] and _q_rope_policy() != _state['q_rope'])):
                _fatal('V41 PCP ownership/policy changed; restart process')
            loaded = sys.modules.get('hcu_vllm_das.models.v41_pcp')
            if loaded is not None:
                loaded.check_native_contract()
            inverse_rope = sys.modules.get('hcu_vllm_das.inverse_rope_pcp')
            if inverse_rope is not None:
                inverse_rope.check_health()
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
        from .layer_split_config import register_cli
        layout = register_cli()
        fp4 = fp4_selected()
        rope_mode = inverse_rope_mode()
        if rope_mode == 'hcu' and not pcp_selected:
            _fatal('Selected inverse RoPE requires the PCP model owner')
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
            profile_worker = _profile_policy() if pcp_selected else None
            if pcp_selected:
                _install_profile_worker()
                if profile_worker or layout is not None:
                    # W4A8 profiling also uses the factory with a replicated KV cache.
                    from .dsa_prefill_runner import install as install_prefill_runner
                    install_prefill_runner()
            if selected and _key(entry) == _NATIVE:
                ModelRegistry.register_model(_ARCH, ':'.join(_PLUGIN))
            entry = ModelRegistry.models.get(_ARCH)
            if _key(entry) != (_PLUGIN if selected else _NATIVE):
                raise RuntimeError('V41 PCP model publication lost its owner')
            if layout is not None:
                layout.register_kv_cache_layout('das-hcu-v41-pcp', _cache_builder, _localize_cache)
            envs.environment_variables[_FACTOR] = policy_factor
            _state = dict(enabled=pcp_selected, host=host_selected, fp4=fp4,
                          inverse_rope_mode=rope_mode,
                          profile_worker=profile_worker,
                          layer_split_layout=layout,
                          q_rope=_q_rope_policy() if pcp_selected else False,
                          registry=ModelRegistry, entry=entry, register=register)
            check_health()
            if pcp_selected:
                _logger.info('DSV4.1 native PCP RoPE policy: inverse=%s query=%s',
                             _state['inverse_rope_mode'], 'hcu' if _state['q_rope'] else 'stock')
        except BaseException as error:
            try:
                _fatal('V41 PCP publication failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def extension_spec():
    capabilities = {'hcu-v41-pcp-eager-attention', 'hcu-v41-private-engram-host'}
    # Keep #399's required-capability contract on the actual PCP owner. The
    # independent Channel-FP8 quantization owner does not imply PCP activation.
    if _flag(_FLAG, default=False):
        capabilities.add('dsv41-pcp-megamoe')
    return ExtensionSpec(name='das-hcu-v41-pcp', api_version=1, priority=116,
        capabilities=frozenset(capabilities),
        activate=activate_extension, check_health=check_health)
