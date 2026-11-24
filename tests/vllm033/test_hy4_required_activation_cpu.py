# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Public Adapter loader contracts; registry and CT prerequisites are doubles."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.extensions import (
    ExtensionCommitError, ExtensionContext, ExtensionStatus,
    RequiredExtensionError, activate_extensions,
)
from hcu_vllm_das import hy4_ct_model_activation as ct_models
from hcu_vllm_das import hy4_frontend as frontend
from hcu_vllm_das import hy4_quant_activation as quant

CT_ENTRY = 'das_hcu_hy4_ct_w4a8'
FRONTEND_ENTRY = 'das_hcu_hy4_frontend'
FLAG = 'VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL'


def context(entry, capability, requirement):
    return ExtensionContext('0.5.22', '0.33.0',
        requested_extensions={entry} if requirement == 'entry' else (),
        requested_capabilities={capability} if requirement == 'capability' else ())


def load(owner, entry, ctx):
    return activate_extensions(ctx,
        entry_points=[NS(name=entry, load=lambda: owner.extension_spec)],
        health_check_sink=lambda report, check: check())


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(ct_models, '_publication', None)
    entries = {
        'HYV4ForCausalLM': NS(module_name='vllm.models.hy_v4', class_name='HYV4ForCausalLM'),
        'HYV4MTPModel': NS(module_name='vllm.models.hy_v4', class_name='HYV4MTP'),
    }
    calls = []

    def register(arch, path):
        calls.append(arch)
        module, cls = path.split(':')
        entries[arch] = NS(module_name=module, class_name=cls)

    value = NS(models=entries, register_model=register, calls=calls)
    monkeypatch.setattr(ct_models, '_registry', lambda: value)
    # Isolate quantization prerequisites, retaining the actual model activation
    # and its health check through the public loader.
    monkeypatch.setattr(quant, 'adapter_module', lambda name:
        NS(set_additional_config_provider=lambda provider: None))
    monkeypatch.setattr(quant, 'check_health', ct_models.check_health)
    return value


@pytest.mark.parametrize('requirement', ['entry', 'capability'])
def test_public_ct_loader_publishes_backbone_and_mtp(registry, requirement):
    ctx = context(CT_ENTRY, 'hy4-ct-w4a8', requirement)
    for _ in range(2):
        reports = load(quant, CT_ENTRY, ctx)
        assert len(reports) == 1 and reports[0].status is ExtensionStatus.ACTIVE
    assert registry.calls == ['HYV4ForCausalLM', 'HYV4MTPModel']
    assert registry.models['HYV4ForCausalLM'].class_name == 'HcuHY4CTForCausalLM'
    assert registry.models['HYV4MTPModel'].class_name == 'HcuHY4CTMTP'


@pytest.mark.parametrize('requirement', ['entry', 'capability'])
@pytest.mark.parametrize('static_requirement', ['entry', 'capability'])
def test_ct_request_still_defers_to_either_static_owner(
        registry, requirement, static_requirement):
    ctx = context(CT_ENTRY, 'hy4-ct-w4a8', requirement)
    ctx = ExtensionContext(ctx.adapter_version, ctx.vllm_version,
        requested_extensions=ctx.requested_extensions | (
            {'das_hcu_hy4_static'} if static_requirement == 'entry' else set()),
        requested_capabilities=ctx.requested_capabilities | (
            {'hy4-static-eplb'} if static_requirement == 'capability' else set()))
    quant.activate(ctx)
    assert registry.calls == []


def test_optional_ct_does_not_publish_models(registry):
    load(quant, CT_ENTRY, ExtensionContext('0.5.22', '0.33.0'))
    assert registry.calls == []


@pytest.fixture
def off_frontend(monkeypatch):
    monkeypatch.setattr(frontend, '_state', None)
    monkeypatch.setenv(FLAG, '0')
    monkeypatch.setattr(frontend, 'import_module', lambda name: pytest.fail(name))
    return frontend


@pytest.mark.parametrize('requirement', ['entry', 'capability'])
@pytest.mark.parametrize('optional_first', [False, True])
def test_public_frontend_loader_rejects_required_off(
        off_frontend, requirement, optional_first):
    if optional_first:
        reports = load(frontend, FRONTEND_ENTRY, ExtensionContext('0.5.22', '0.33.0'))
        assert reports[0].status is ExtensionStatus.ACTIVE
        frontend.check_health()
    previous = frontend._state
    ctx = context(FRONTEND_ENTRY, 'bytelevel-prompt-tail', requirement)
    with pytest.raises(RequiredExtensionError) as error:
        load(frontend, FRONTEND_ENTRY, ctx)
    assert len(error.value.reports) == 1
    assert error.value.reports[0].status is ExtensionStatus.FAILED
    assert 'requires ' + FLAG + '=1' in error.value.reports[0].reason
    assert frontend._state is previous


@pytest.mark.parametrize('requirement', ['entry', 'capability'])
@pytest.mark.parametrize('drift', ['off', 'invalid', 'owner'])
def test_required_frontend_retains_commit_fatal_drift_priority(
        monkeypatch, requirement, drift):
    monkeypatch.setattr(frontend, '_state', None)
    monkeypatch.setenv(FLAG, '1')

    class Detokenizer:
        def __init__(self, tokenizer, request):
            pass

    module = NS(FastIncrementalDetokenizer=Detokenizer)
    monkeypatch.setattr(frontend, 'import_module', lambda name: module)
    load(frontend, FRONTEND_ENTRY, ExtensionContext('0.5.22', '0.33.0'))
    if drift == 'owner':
        Detokenizer.__init__ = lambda *args: None
    else:
        monkeypatch.setenv(FLAG, '0' if drift == 'off' else 'invalid')
    with pytest.raises(ExtensionCommitError):
        load(frontend, FRONTEND_ENTRY,
             context(FRONTEND_ENTRY, 'bytelevel-prompt-tail', requirement))


@pytest.mark.parametrize('requirement', ['entry', 'capability'])
def test_required_on_frontend_reentry_keeps_constructor(monkeypatch, requirement):
    monkeypatch.setattr(frontend, '_state', None)
    monkeypatch.setenv(FLAG, '1')

    class Detokenizer:
        def __init__(self, tokenizer, request):
            pass

    module = NS(FastIncrementalDetokenizer=Detokenizer)
    monkeypatch.setattr(frontend, 'import_module', lambda name: module)
    ctx = context(FRONTEND_ENTRY, 'bytelevel-prompt-tail', requirement)
    load(frontend, FRONTEND_ENTRY, ctx)
    constructor = Detokenizer.__init__
    load(frontend, FRONTEND_ENTRY, ctx)
    assert Detokenizer.__init__ is constructor
