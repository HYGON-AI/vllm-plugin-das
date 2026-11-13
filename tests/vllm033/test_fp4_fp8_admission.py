# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP8 cache admission through real packer/indexer owners and loader, CPU only."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import fp4_indexer, hcu_config, hcu_env
from hcu_adapter.extensions import ExtensionContext, ExtensionStatus, activate_extensions


@pytest.fixture
def owners(monkeypatch):
    from vllm import config, envs, platforms
    from vllm.model_executor import custom_op
    from hcu_vllm_das import v41_indexer_activation as activation, fp4_packer, v41_indexer

    monkeypatch.setattr(hcu_config, '_STATE', {
        'loaded': True, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    for owner in (activation, fp4_packer):
        for key, value in [('_state', None), ('_failure', None), ('_publishing', False)]:
            monkeypatch.setattr(owner, key, value)
    monkeypatch.setattr(platforms, 'current_platform', NS(
        is_rocm=lambda: True, device_name='hcu'))
    monkeypatch.setattr(hcu_env, 'arch_name', lambda: 'gfx936')
    monkeypatch.setattr(envs, 'environment_variables', dict(envs.environment_variables))
    envs.environment_variables.pop(activation._FACTOR, None)
    monkeypatch.setattr(custom_op, 'op_registry_oot', {})
    monkeypatch.setattr(v41_indexer, 'register_op', lambda: None)
    monkeypatch.delenv(fp4_indexer.VARIABLE, raising=False)
    monkeypatch.delenv('HCU_VLLM_FP4_PACKER_IMPL', raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    cfg = NS(attention_config=NS(resolve_indexer_kv_dtype=lambda default: 'fp8'))
    monkeypatch.setattr(config, 'get_current_vllm_config_or_none', lambda: cfg)
    return NS(indexer=activation, packer=fp4_packer, cfg=cfg, scorer=v41_indexer,
              registry=custom_op.op_registry_oot)


def load(owners):
    points = [NS(name=name, load=lambda owner=owner: owner.extension_spec)
              for name, owner in [('indexer', owners.indexer), ('packer', owners.packer)]]
    return activate_extensions(ExtensionContext('0.1.0', '0.33.0'), entry_points=points)


@pytest.mark.parametrize('packer_mode', ['stock', 'default-gfx936'])
def test_fp8_publishes_without_fp4_preflight(owners, monkeypatch, packer_mode):
    if packer_mode == 'stock':
        monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    assert fp4_indexer.selected()  # Default knob alone is on in both deployments.
    reports = load(owners)
    indexer = next(r for r in reports if r.entry_point_name == 'indexer')
    assert indexer.status is ExtensionStatus.ACTIVE, indexer.reason
    assert owners.indexer._state['fp4'] is False
    assert owners.registry['SparseAttnIndexer'] is owners.scorer.HcuV41SparseAttnIndexer
    owners.indexer.check_health()
    # Turning off an irrelevant FP4 knob must not invalidate an FP8 deployment.
    monkeypatch.setenv(fp4_indexer.VARIABLE, 'stock')
    owners.indexer.check_health()
    assert next(r for r in load(owners) if r.entry_point_name == 'indexer').status is ExtensionStatus.ACTIVE


def test_mxfp4_still_requires_published_lowering(owners, monkeypatch):
    owners.cfg.attention_config.resolve_indexer_kv_dtype = lambda default: 'mxfp4'
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    reports = load(owners)
    report = next(r for r in reports if r.entry_point_name == 'indexer')
    assert report.status is ExtensionStatus.FAILED
    assert owners.indexer._state is None
    assert 'SparseAttnIndexer' not in owners.registry


def test_fp8_to_mxfp4_policy_change_latches(owners, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    reports = load(owners)
    assert next(r for r in reports if r.entry_point_name == 'indexer').status is ExtensionStatus.ACTIVE
    owners.cfg.attention_config.resolve_indexer_kv_dtype = lambda default: 'mxfp4'
    with pytest.raises(ExtensionCommitError):
        owners.indexer.check_health()
    owners.cfg.attention_config.resolve_indexer_kv_dtype = lambda default: 'fp8'
    with pytest.raises(ExtensionCommitError):
        owners.indexer.check_health()


@pytest.mark.parametrize('dtype', ['fp8', 'mxfp4'])
def test_plugin_load_before_model_config_is_available(owners, monkeypatch, dtype):
    from vllm import config
    from hcu_adapter.extensions import ExtensionCommitError
    monkeypatch.setattr(config, 'get_current_vllm_config_or_none', lambda: None)
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    reports = load(owners)
    report = next(r for r in reports if r.entry_point_name == 'indexer')
    assert report.status is ExtensionStatus.ACTIVE, report.reason
    assert owners.indexer._state['fp4'] is None
    owners.indexer.check_health()  # Host health runs outside model context too.
    owners.cfg.attention_config.resolve_indexer_kv_dtype = lambda default: dtype
    if dtype == 'fp8':
        owners.indexer.admit_config(owners.cfg)
        assert owners.indexer._state['fp4'] is False
        owners.indexer.admit_config(owners.cfg)
        owners.indexer.check_health()
    else:
        with pytest.raises(ExtensionCommitError):
            owners.indexer.admit_config(owners.cfg)
        with pytest.raises(ExtensionCommitError):
            owners.indexer.check_health()


def test_indexer_constructor_resolves_config_before_native_allocation(owners, monkeypatch):
    from vllm import config
    monkeypatch.setattr(config, 'get_current_vllm_config_or_none', lambda: None)
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    load(owners)
    owners.cfg.model_config = NS(hf_config=NS(model_type='deepseek_v41'))
    monkeypatch.setattr(owners.scorer, 'current_config', lambda: owners.cfg)
    calls = []
    base = owners.scorer.HcuV41SparseAttnIndexer.__bases__[0]

    def construct(self):
        assert owners.indexer._state['fp4'] is False
        calls.append('native-allocation')

    monkeypatch.setattr(base, '__init__', construct)
    owners.scorer.HcuV41SparseAttnIndexer()
    assert calls == ['native-allocation']


def test_pending_mxfp4_admission_keeps_provider_and_digest_checks(owners, monkeypatch):
    from vllm import config
    from hcu_adapter import registry
    from hcu_adapter.extensions import ExtensionCommitError
    from hcu_vllm_das import fp4_indexer as native, fp4_indexer_lightop as reader
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.setattr(config, 'get_current_vllm_config_or_none', lambda: None)
    monkeypatch.setenv('HCU_VLLM_FP4_PACKER_IMPL', 'stock')
    load(owners)
    owners.cfg.attention_config.resolve_indexer_kv_dtype = lambda default: 'mxfp4'
    # Hardware/provider availability are explicit doubles. The real admission,
    # record publication, digest comparison and failure latch still execute.
    calls = []
    monkeypatch.setattr(owners.packer, '_state', {'mode': 'triton'})
    monkeypatch.setattr(owners.packer, 'check_health', lambda: calls.append('packer'))
    monkeypatch.setattr(reader, 'load_providers', lambda: calls.append('providers'))
    monkeypatch.setattr(native, 'check_native_contract', lambda: calls.append('native'))
    monkeypatch.setattr(owners.indexer, '_compute_digest', lambda: 'digest')
    monkeypatch.setattr(owners.indexer, '_digest_signature', lambda: ('signature',))
    owners.indexer.admit_config(owners.cfg)
    assert owners.indexer._state['fp4'] is True
    assert owners.indexer.policy_factor() == ('fp4-lightop', 'digest')
    assert registry.get('vllm_fp4_indexer')['implementation'] == 'lightop'
    assert {'packer', 'providers', 'native'} <= set(calls)
    owners.indexer.check_health()  # Works after leaving a model config context.
    monkeypatch.setattr(owners.indexer, '_digest_signature', lambda: ('changed',))
    monkeypatch.setattr(owners.indexer, '_compute_digest', lambda: 'changed')
    with pytest.raises(ExtensionCommitError, match='implementation changed'):
        owners.indexer.check_health()
