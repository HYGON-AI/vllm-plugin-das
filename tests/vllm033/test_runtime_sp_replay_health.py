# SPDX-License-Identifier: Apache-2.0
"""Required-provider model health without executing a device graph or model."""
import gc
import os
import signal
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_runtime_sp_attention import core, sp


@pytest.fixture
def owners(core, monkeypatch):
    from hcu_adapter.bridges import vllm as bridge
    from hcu_adapter.extensions import ExtensionContext
    from hcu_vllm_das.parallel import runtime_sp_activation as activation
    from hcu_vllm_das.parallel import runtime_sp_models as models

    class Platform:
        selector = None
        @classmethod
        def register_device_communicator_selector(cls, selector):
            cls.selector = selector
        @classmethod
        def owns_device_communicator_selector(cls, selector):
            return cls.selector is selector

    monkeypatch.setattr(activation, '_platform', None)
    monkeypatch.setattr(activation, '_model_health', ())
    monkeypatch.setattr(activation, '_lifecycle', activation.ProcessLocalLifecycle(
        runtime_action=activation._ACTION, error_type=activation.ExtensionCommitError))
    monkeypatch.setattr(bridge, 'HcuPlatform', Platform)
    # This is the activation ABI prerequisite, not a device/Worker execution
    # claim. The real Adapter97 Worker/Graph composition is qualified separately.
    if not callable(getattr(bridge, 'require_forward_healthy', None)):
        monkeypatch.setattr(bridge, 'require_forward_healthy', lambda: None, raising=False)
    context = ExtensionContext(adapter_version='0.5.22', vllm_version='0.33.0',
        requested_capabilities=frozenset({activation.CAPABILITY}))
    activation.activate(context)
    core.device_communicator = SimpleNamespace(all_to_all_single=lambda out, value: out.copy_(value))
    monkeypatch.setattr(models, 'get_tp_group', lambda: core)
    monkeypatch.setattr(models, 'custom_sp_requested', lambda config: True)
    for name in ('install_attention_layers', 'install_moe_factory_policy',
                 'check_attention_layers', 'check_moe_factory_policy'):
        monkeypatch.setattr(models, name, lambda: None)

    class Base(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            self.projection = sp.RuntimeSPQKVParallelLinear(16, 4, 8, 2,
                bias=False, prefix='model.layers.0.self_attn.qkv_proj')

    class Model(models.RuntimeSPModelBoundary, Base):
        pass

    return SimpleNamespace(activation=activation, models=models, Model=Model,
        group=core, platform=Platform, bridge=bridge, context=context)


def test_model_constructor_binds_health_without_retaining_weights(owners):
    model = owners.Model(vllm_config=None)
    owners.activation.check_health()
    assert len(owners.activation._model_health) == 1
    assert owners.activation._model_health[0][0]() is model
    del model
    gc.collect()
    with pytest.raises(RuntimeError, match='owner expired'):
        owners.activation.check_health()


@pytest.mark.parametrize('fault', ['model_group', 'projection_group', 'projection_binding',
                                  'projection_fields', 'rank', 'registry'])
def test_health_refuses_owner_drift_before_cached_replay(owners, monkeypatch, fault):
    model = owners.Model(vllm_config=None)
    replay = []
    def execute_cached():
        owners.activation.check_health()
        replay.append(True)
    execute_cached()
    assert replay == [True]
    replay.clear()
    foreign = SimpleNamespace(world_size=4, rank_in_group=0)
    if fault == 'model_group':
        monkeypatch.setattr(owners.models, 'get_tp_group', lambda: foreign)
    elif fault == 'projection_group':
        monkeypatch.setattr(sp, 'get_tp_group', lambda: foreign)
    elif fault == 'projection_binding':
        model.projection = nn.Identity()
    elif fault == 'projection_fields':
        model.projection._runtime_sp_size = 8
    elif fault == 'rank':
        owners.group.rank_in_group = 1
    else:
        monkeypatch.setattr(owners.models, 'check_attention_layers',
            lambda: (_ for _ in ()).throw(RuntimeError('foreign OOT owner')))
    with pytest.raises(RuntimeError):
        execute_cached()
    assert replay == []


def test_model_registration_is_idempotent(owners):
    model = owners.Model(vllm_config=None)
    owners.activation.register_model_health(model)
    assert len(owners.activation._model_health) == 1


def test_activation_refuses_missing_adapter_replay_boundary(owners, monkeypatch):
    activation = owners.activation
    monkeypatch.setattr(activation, '_platform', None)
    monkeypatch.setattr(activation, '_lifecycle', activation.ProcessLocalLifecycle(
        runtime_action=activation._ACTION, error_type=activation.ExtensionCommitError))
    owners.platform.selector = None
    monkeypatch.delattr(owners.bridge, 'require_forward_healthy')
    with pytest.raises(RuntimeError, match='cached FULL replay'):
        activation.activate(owners.context)
    assert owners.platform.selector is None


def test_replay_health_fork_refuses_without_inherited_lock_deadlock(owners):
    model = owners.Model(vllm_config=None)
    locked = threading.Event()
    release = threading.Event()
    def hold_parent_lock():
        with owners.activation._lifecycle._lock:
            locked.set()
            release.wait(15)
    thread = threading.Thread(target=hold_parent_lock)
    thread.start()
    assert locked.wait(5)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        signal.alarm(10)
        try:
            owners.activation.check_health()
        except RuntimeError as error:
            os.write(write_fd, str(error).encode())
            os._exit(0)
        os._exit(1)
    os.close(write_fd)
    try:
        message = os.read(read_fd, 4096)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert b'process or TP ownership changed' in message
    finally:
        release.set()
        thread.join(5)
        os.close(read_fd)
    owners.activation.check_health()
