# SPDX-License-Identifier: Apache-2.0
"""Construction admission and rollback for the owned HY4 HT expert gather."""
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
import sys

import pytest


def scope():
    spec = importlib.util.spec_from_file_location('gather_ht_scope',
        Path(__file__).with_name('test_hy4_ht_int8.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.scope()


def setup(monkeypatch):
    policy = importlib.import_module('hcu_vllm_das.hy4_expert_gather')
    runtime = importlib.import_module('hcu_vllm_das.moe.ct_w4a8_runtime')
    ht = importlib.import_module('hcu_vllm_das.hy4_ht_int8')
    monkeypatch.setattr(ht, '_device_arch', lambda: 'gfx938')
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('HCU_VLLM_HY4_EXPERT_GATHER', '1')
    expert = object.__new__(runtime.HcuCTContiguousExperts)
    expert._gather = runtime.deepgemm_unpermute_and_reduce
    vendor = ModuleType('hcu_adapter.bridges.vllm_expert_gather')
    calls = []
    vendor.unpermute_and_reduce = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, vendor.__name__, vendor)
    return policy, runtime, expert, calls


def test_admitted_ht_uses_bridge_and_rollback_retains_native(monkeypatch):
    policy, runtime, expert, calls = setup(monkeypatch)
    native_calls = []
    native = lambda *args, **kwargs: native_calls.append((args, kwargs))
    monkeypatch.setattr(runtime, 'deepgemm_unpermute_and_reduce', native)
    expert._gather = native
    cfg, moe, prepare = scope()
    policy.configure(expert, cfg, moe, prepare)
    args = tuple(object() for _ in range(6))
    expert._gather(*args)
    assert calls == [(args, {})] and native_calls == []
    monkeypatch.setenv('HCU_VLLM_HY4_EXPERT_GATHER', '0')
    expert._gather(*args)
    assert native_calls == [(args, {})] and len(calls) == 1


@pytest.mark.parametrize('kind', ['off','master','model','dp','ll','other_expert','foreign'])
def test_unqualified_scope_keeps_function_identity(monkeypatch, kind):
    policy, runtime, expert, calls = setup(monkeypatch)
    cfg, moe, prepare = scope()
    if kind == 'off': monkeypatch.delenv('HCU_VLLM_HY4_EXPERT_GATHER')
    elif kind == 'master': monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    elif kind == 'model': cfg.model_config.hf_config.model_type = 'other'
    elif kind == 'dp': cfg.parallel_config.data_parallel_size = 16
    elif kind == 'll': moe.moe_parallel_config.all2all_backend = 'deepep_low_latency'
    elif kind == 'other_expert': expert = type('Other', (), {'_gather': staticmethod(lambda: None)})()
    elif kind == 'foreign': expert._gather = lambda: None
    before = expert._gather
    policy.configure(expert, cfg, moe, prepare)
    assert expert._gather is before and calls == []
