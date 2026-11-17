# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Discovery checks only the binder; provider contracts are activation-time."""
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
if {missing_binder!r}:
    module = importlib.import_module('hcu_adapter.{adapter_owner}')
    del module.set_provider_binder
try:
    facade = importlib.import_module('hcu_vllm_das.{owner}')
except AdapterContractError as error:
    assert {missing_binder!r}, str(error)
    assert 'hcu_adapter.{adapter_owner}.set_provider_binder' in str(error)
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
    from hcu_adapter import dsv41_channel_fp8
    monkeypatch.setattr(_seams, '_bound', set())
    monkeypatch.delattr(dsv41_channel_fp8, 'set_moe_provider')
    for bind in (_seams.bind_dsv41, _seams.bind_ct_w4a8):
        with pytest.raises(_seams.AdapterContractError, match='set_moe_provider'):
            bind()
    assert not _seams._bound
