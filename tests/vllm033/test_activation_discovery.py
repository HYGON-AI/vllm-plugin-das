# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Discovery checks only the binder; provider contracts are activation-time."""

from hcu_vllm_das.adapter_api import adapter_module as _adapter_module
import subprocess
import sys

import pytest


@pytest.mark.parametrize('owner', ['dsv41_activation', 'ct_w4a8_activation'])
@pytest.mark.parametrize('missing_binder', [False, True])
def test_discovery_is_framework_free_and_names_missing_binder(owner, missing_binder):
    adapter_owner = ('ct_w4a8_registration' if owner == 'ct_w4a8_activation'
                     else 'dsv41_activation')
    code = f'''
import sys
sys.path[:0] = {list(sys.path)!r}
import importlib
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'vllm', 'torch', 'aiter', 'lightop', 'vllm_hcu'}}:
            raise AssertionError('discovery imported ' + fullname)
sys.meta_path.insert(0, Block())
from hcu_vllm_das._seams import AdapterContractError
from hcu_vllm_das.adapter_api import adapter_module, adapter_module_name
qualified_owner = adapter_module_name({adapter_owner!r})
if {missing_binder!r}:
    module = adapter_module({adapter_owner!r})
    del module.set_provider_binder
try:
    facade = importlib.import_module('hcu_vllm_das.{owner}')
except AdapterContractError as error:
    assert {missing_binder!r}, str(error)
    assert qualified_owner + '.set_provider_binder' in str(error)
else:
    assert not {missing_binder!r}, 'missing binder was accepted'
    assert facade.extension_spec().check_health is not None
assert not set(sys.modules).intersection({{'torch', 'vllm', 'aiter', 'lightop', 'vllm_hcu'}})
'''
    result = subprocess.run([sys.executable, '-S', '-B', '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_provider_contract_is_still_checked_at_activation(monkeypatch):
    from hcu_vllm_das import _seams
    # Providers can now be imported inside the framework, but missing seams
    # must still refuse with their name before any provider is bound.
    dsv41_channel_fp8 = _adapter_module('dsv41_channel_fp8')
    monkeypatch.setattr(_seams, '_bound', set())
    monkeypatch.delattr(dsv41_channel_fp8, 'set_moe_provider')
    for bind in (_seams.bind_dsv41, _seams.bind_ct_w4a8):
        with pytest.raises(_seams.AdapterContractError, match='set_moe_provider'):
            bind()
    assert not _seams._bound


@pytest.mark.parametrize('owner', ['hy4_frontend', 'hy4_quant_activation'])
def test_hy4_discovery_does_not_import_framework_or_vendor(owner):
    code = f'''
import sys, importlib, importlib.abc
sys.path[:0] = {list(sys.path)!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'vllm', 'torch', 'aiter', 'lightop', 'deepgemm', 'tokenizers'}}:
            raise AssertionError('discovery imported ' + fullname)
sys.meta_path.insert(0, Block())
module = importlib.import_module('hcu_vllm_das.{owner}')
assert callable(module.extension_spec().activate)
'''
    result = subprocess.run([sys.executable, '-S', '-B', '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('registration_enabled', [True, False])
def test_hy4_capability_waits_for_ct_registration(monkeypatch, registration_enabled):
    """Use the real dispatcher and declared priorities; isolate registry I/O."""
    from dataclasses import replace
    from types import SimpleNamespace

    from hcu_adapter.extensions import (
        ExtensionContext, RequiredExtensionError, activate_extensions,
    )
    from hcu_vllm_das import hy4_quant_activation as hy4

    ct = _adapter_module('ct_w4a8_registration')
    calls = []

    def register(context):
        if not registration_enabled:
            raise RuntimeError('CT registration disabled')
        calls.append('registered')

    def health():
        if calls != ['registered']:
            raise RuntimeError('CT W4A8 registration is not active')

    monkeypatch.setattr(hy4, 'check_health', health)
    specs = {
        'ct': replace(ct.extension_spec(), activate=register),
        'hy4': hy4.extension_spec(),
    }
    # Reverse input order: sorting, rather than discovery order, must enforce
    # the dependency. Existing CT publication/kill tests cover its transaction.
    points = [SimpleNamespace(name=n, load=lambda n=n: lambda: specs[n])
              for n in ('hy4', 'ct')]
    context = ExtensionContext('0.5.22', '0.33.0',
        requested_capabilities=frozenset({'hy4-ct-w4a8'}))
    if registration_enabled:
        reports = activate_extensions(context, entry_points=points)
        assert all(report.status.value == 'active' for report in reports)
    else:
        with pytest.raises(RequiredExtensionError, match='hy4-ct-w4a8'):
            activate_extensions(context, entry_points=points)
