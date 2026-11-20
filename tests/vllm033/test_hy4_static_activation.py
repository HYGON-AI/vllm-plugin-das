# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Adapter lifecycle + public model registry contract, no model execution."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext, activate_extensions
from hcu_adapter.infra.runtime_lifecycle import ProcessLocalLifecycle
from hcu_vllm_das import hy4_static_activation as owner


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(owner, '_publication', None)
    entries = {
        'HYV4ForCausalLM': NS(module_name='vllm.models.hy_v4', class_name='HYV4ForCausalLM'),
        'HYV4MTPModel': NS(module_name='vllm.models.hy_v4', class_name='HYV4MTP')}
    calls = []
    def register(arch, path):
        calls.append(arch)
        module, cls = path.split(':')
        entries[arch] = NS(module_name=module, class_name=cls)
    registry = NS(models=entries, register_model=register, calls=calls)
    monkeypatch.setattr(owner, '_registry', lambda: registry)
    from hcu_vllm_das import hy4_quant_activation
    monkeypatch.setattr(hy4_quant_activation, 'check_health', lambda: None)
    return registry


def context():
    return ExtensionContext('0.5.22', '0.33.0',
                            requested_capabilities=frozenset({'hy4-static-eplb'}))


def test_unrequested_provider_does_not_import_framework_or_register(monkeypatch):
    monkeypatch.setattr(owner, '_registry', lambda: pytest.fail('Core import'))
    with pytest.raises(RuntimeError, match='explicit'):
        owner.activate(ExtensionContext('0.5.22', '0.33.0'))


def test_both_models_are_published_lazily_and_reactivation_is_idempotent(registry):
    owner.activate(context())
    assert registry.calls == ['HYV4ForCausalLM', 'HYV4MTPModel']
    owner.check_health()
    first = dict(registry.models)
    owner.activate(context())
    assert registry.models == first
    assert len(registry.calls) == 2


def test_foreign_mtp_owner_is_rejected_before_backbone_publication(registry):
    registry.models['HYV4MTPModel'] = NS(module_name='foreign', class_name='Draft')
    with pytest.raises(RuntimeError, match='owner'):
        owner.activate(context())
    assert registry.calls == []


def test_partial_publication_signals_commit_fatal(registry):
    original = registry.register_model
    def fail_second(arch, path):
        if arch == 'HYV4MTPModel':
            raise RuntimeError('second publication failed')
        original(arch, path)
    registry.register_model = fail_second
    with pytest.raises(ExtensionCommitError):
        owner.activate(context())
    assert registry.calls == ['HYV4ForCausalLM']


def test_adapter_owns_persistent_health_failure(registry):
    host = ProcessLocalLifecycle(runtime_action='runtime', error_type=ExtensionCommitError)
    point = NS(name='das_hcu_hy4_static', load=lambda: owner.extension_spec)
    def activate():
        reports = activate_extensions(context(), entry_points=[point],
            health_check_sink=host.record_extension_health)
        host.record_extension_reports(reports)
    host.run_once('runtime', activate)
    host.require_extension_healthy('hy4-static-eplb')
    saved = registry.models['HYV4MTPModel']
    registry.models['HYV4MTPModel'] = NS(module_name='foreign', class_name='Draft')
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('hy4-static-eplb')
    registry.models['HYV4MTPModel'] = saved
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('hy4-static-eplb')
