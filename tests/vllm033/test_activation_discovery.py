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


@pytest.mark.parametrize('legacy,canonical,expected', [
    (None, None, False), ('0', None, False), ('false', None, False),
    ('1', None, True), ('TrUe', None, True),
    ('1', '0', False), ('1', '1', True), ('false', '1', True),
])
def test_mooncake_alias_uses_core_policy_without_wrapping_outputs(legacy, canonical, expected):
    """Exercise the real Core environment owner, including explicit quick-off."""
    code = f'''
import sys,os
sys.path[:0] = {list(sys.path)!r}
from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import mooncake_trace
for key,value in [(mooncake_trace.LEGACY_FLAG,{legacy!r}), (mooncake_trace.CORE_FLAG,{canonical!r})]:
    os.environ.pop(key,None)
    if value is not None:
        os.environ[key]=value
from vllm import envs
assert not envs._is_envs_cache_enabled()
getter=envs.environment_variables[mooncake_trace.CORE_FLAG]
before=set(sys.modules)
spec=mooncake_trace.extension_spec()
context=ExtensionContext(adapter_version='0.5.22',vllm_version='0.33.0')
spec.activate(context)
spec.activate(context)
spec.check_health()
assert envs.VLLM_MOONCAKE_TTFT_TRACE is {expected!r}
assert envs.environment_variables[mooncake_trace.CORE_FLAG] is getter
assert not envs._is_envs_cache_enabled()
assert os.environ.get(mooncake_trace.CORE_FLAG) == {canonical if canonical is not None else ('1' if expected else None)!r}
assert not any(name == 'torch' or name.startswith(('mooncake','vllm.v1.engine.output_processor', 'vllm.distributed.kv_transfer')) for name in set(sys.modules)-before)
envs.enable_envs_cache()
spec.check_health()
assert envs.VLLM_MOONCAKE_TTFT_TRACE is {expected!r}
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mooncake_alias_refuses_late_cached_enablement():
    """Keep the real cached setting intact and do not replace its getter."""
    code = f'''
import sys,os
sys.path[:0] = {list(sys.path)!r}
from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import mooncake_trace
os.environ.pop(mooncake_trace.CORE_FLAG,None)
os.environ[mooncake_trace.LEGACY_FLAG]='1'
from vllm import envs
getter=envs.environment_variables[mooncake_trace.CORE_FLAG]
envs.enable_envs_cache()
cached=envs.__getattr__
assert envs.VLLM_MOONCAKE_TTFT_TRACE is False
try:
    mooncake_trace.extension_spec().activate(ExtensionContext('0.5.22','0.33.0'))
except RuntimeError as exc:
    assert 'after Core cached' in str(exc)
else:
    raise AssertionError('late legacy enablement was accepted')
assert os.environ.get(mooncake_trace.CORE_FLAG) is None
assert envs.VLLM_MOONCAKE_TTFT_TRACE is False
assert envs.environment_variables[mooncake_trace.CORE_FLAG] is getter
assert envs.__getattr__ is cached
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('changed', ['VLLM_HCU_MOONCAKE_TTFT_TRACE', 'VLLM_MOONCAKE_TTFT_TRACE'])
def test_mooncake_alias_health_rejects_policy_drift(changed):
    code = f'''
import sys,os
sys.path[:0] = {list(sys.path)!r}
from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import mooncake_trace
for key in [mooncake_trace.LEGACY_FLAG,mooncake_trace.CORE_FLAG]:
    os.environ.pop(key,None)
spec=mooncake_trace.extension_spec()
spec.activate(ExtensionContext('0.5.22','0.33.0'))
os.environ[{changed!r}]='1'
try:
    spec.check_health()
except RuntimeError as exc:
    assert 'policy changed' in str(exc)
else:
    raise AssertionError('policy drift was silently accepted')
assert os.environ[{changed!r}]=='1'
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mooncake_alias_activates_through_installed_required_owner():
    """Use the real wheel entry point and Adapter's required-owner API."""
    code = f'''
import sys,os
sys.path[:0] = {list(sys.path)!r}
from importlib.metadata import entry_points
from hcu_adapter.extensions import ExtensionContext,ExtensionStatus,activate_extensions
from vllm.v1.engine.output_processor import OutputProcessor
from vllm import envs
original=OutputProcessor.process_outputs
os.environ['VLLM_HCU_MOONCAKE_TTFT_TRACE']='true'
os.environ.pop('VLLM_MOONCAKE_TTFT_TRACE',None)
point,=[p for p in entry_points(group='hcu_adapter.extensions') if p.name=='das_hcu_mooncake_trace']
health=[]
context=ExtensionContext('0.5.22','0.33.0',requested_extensions=frozenset({{'das_hcu_mooncake_trace'}}))
reports=activate_extensions(context,[point],health_check_sink=lambda report,check: health.append(check))
assert len(reports)==1 and reports[0].status is ExtensionStatus.ACTIVE
assert reports[0].capabilities==frozenset({{'mooncake-ttft-trace-env'}})
assert len(health)==1
envs.enable_envs_cache()
health[0]()
assert envs.VLLM_MOONCAKE_TTFT_TRACE is True
assert OutputProcessor.process_outputs is original
assert not any(n.startswith('mooncake') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('missing_api', [False, True])
def test_mooncake_alias_required_owner_refuses_unusable_core(missing_api):
    """Keep optional diagnostics and required startup failures distinguishable."""
    code = f'''
import sys,os
sys.path[:0] = {list(sys.path)!r}
from importlib.metadata import entry_points
from hcu_adapter.extensions import (
    ExtensionContext,ExtensionStatus,RequiredExtensionError,activate_extensions,
)
from vllm import envs
from hcu_vllm_das import mooncake_trace
os.environ.pop(mooncake_trace.CORE_FLAG,None)
os.environ[mooncake_trace.LEGACY_FLAG]='1'
getter=envs.environment_variables[mooncake_trace.CORE_FLAG]
if {missing_api!r}:
    del envs.environment_variables[mooncake_trace.CORE_FLAG]
    expected="requires Core's VLLM_MOONCAKE_TTFT_TRACE API"
else:
    envs.enable_envs_cache()
    assert envs.VLLM_MOONCAKE_TTFT_TRACE is False
    expected='after Core cached its environment'
cached=envs.__getattr__
point,=[p for p in entry_points(group='hcu_adapter.extensions') if p.name=='das_hcu_mooncake_trace']
reports=activate_extensions(ExtensionContext('0.5.22','0.33.0'),[point])
assert len(reports)==1 and reports[0].status is ExtensionStatus.FAILED
assert expected in reports[0].reason
try:
    activate_extensions(ExtensionContext('0.5.22','0.33.0',requested_extensions=frozenset({{'das_hcu_mooncake_trace'}})),[point])
except RequiredExtensionError as exc:
    assert len(exc.reports)==1 and exc.reports[0].status is ExtensionStatus.FAILED
    assert expected in exc.reports[0].reason
else:
    raise AssertionError('required tracing owner accepted an unusable Core')
assert os.environ.get(mooncake_trace.CORE_FLAG) is None
assert envs.__getattr__ is cached
if not {missing_api!r}:
    assert envs.VLLM_MOONCAKE_TTFT_TRACE is False
    assert envs.environment_variables[mooncake_trace.CORE_FLAG] is getter
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
