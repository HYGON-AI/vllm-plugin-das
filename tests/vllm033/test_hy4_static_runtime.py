# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared static runtime scope dispatch; no native/device execution."""
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from hcu_vllm_das import static_eplb_runtime as runtime
from test_hy4_static_scope import config


@pytest.fixture
def cfg(monkeypatch):
    cfg = config()
    cfg.additional_config = {'hcu_static_eplb': {
        'expert_map_path': '/test/map.json', 'num_redundant_experts': 16}}
    cfg.kernel_config = NS(moe_backend='deep_gemm')
    cfg.model_config.hf_config.n_routed_experts = 256
    p = cfg.parallel_config
    p.enable_expert_parallel = True
    p.enable_eplb = False
    p.enable_elastic_ep = False
    p.expert_placement_strategy = 'linear'
    p.all2all_backend = 'deepep_high_throughput'
    p.enable_dbo = False
    # The legacy target must not accidentally be widened into HY4 admission.
    legacy = ModuleType('hcu_vllm_das.v41_pcp')
    legacy.validate_config = lambda cfg: False
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    return cfg


def test_hy4_uses_separate_scope_and_active_model_capability(cfg, monkeypatch):
    from hcu_adapter.bridges import vllm as host
    calls = []
    monkeypatch.setattr(host, 'require_extension_healthy', calls.append)
    assert runtime.validate_config(cfg)
    assert calls == ['hy4-static-eplb']


@pytest.mark.parametrize('layer', [3, 77, 78])
def test_both_backbone_and_mtp_allocate_physical_slots_before_load(cfg, monkeypatch, layer):
    from hcu_adapter.bridges import vllm as host
    monkeypatch.setattr(host, 'require_extension_healthy', lambda name: None)
    assert runtime.moe_eplb_kwargs(cfg, f'model.layers.{layer}.mlp') == {
        'enable_eplb': True, 'num_redundant_experts': 16}
    assert runtime.moe_eplb_kwargs(cfg, 'model.layers.79.mlp') == {}


def test_missing_model_owner_cannot_advertise_static_eplb_support(cfg, monkeypatch):
    from hcu_adapter.bridges import vllm as host
    def absent(name):
        raise RuntimeError('missing active ' + name)
    monkeypatch.setattr(host, 'require_extension_healthy', absent)
    with pytest.raises(RuntimeError, match='missing active hy4-static-eplb'):
        runtime.validate_config(cfg)


def test_v41_admission_is_still_delegated_to_its_own_validator(cfg, monkeypatch):
    from hcu_adapter.bridges import vllm as host
    monkeypatch.setattr(host, 'require_extension_healthy', lambda name: pytest.fail('HY4 owner used'))
    cfg.model_config.hf_config.model_type = 'deepseek_v41'
    with pytest.raises(runtime.StaticEplbError, match='V41'):
        runtime.validate_config(cfg)
