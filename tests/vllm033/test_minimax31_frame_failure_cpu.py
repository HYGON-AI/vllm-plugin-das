# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Failed frames release host ownership but never reuse uncertain device state."""
import threading
from types import SimpleNamespace
import pytest
import torch
from hcu_vllm_das.parallel import minimax31_static_runtime as runtime
from hcu_vllm_das.parallel import minimax31_pcp as pcp
from hcu_vllm_das.parallel.minimax31_static_contract import request_base


@pytest.fixture
def batch(monkeypatch):
    key = '00' * 32
    monkeypatch.setattr(runtime, '_FRAME_LOCK', threading.Lock())
    monkeypatch.setattr(runtime, '_FRAME_POISONED', False)
    monkeypatch.setattr(runtime, '_LAYOUT', None)
    monkeypatch.setattr(runtime, '_LAST_LAYER', 4)
    monkeypatch.setattr(runtime, '_REQUESTS', {'request': (key, request_base(key))})
    return SimpleNamespace(num_tokens=2, num_tokens_after_padding=2,
        num_scheduled_tokens=torch.tensor([2]), req_ids=['request'],
        num_computed_tokens_np=torch.tensor([0]), positions=torch.tensor([0, 1]))


@pytest.mark.parametrize('entry', ['execute_model', '_dummy_run'])
@pytest.mark.parametrize('completed', [0, 1, 2])
def test_actual_runner_boundary_cleans_and_latches_failure(monkeypatch, batch, entry, completed):
    calls = []
    failure = RuntimeError('injected model error')
    def fail(self, *args, **kwargs):
        calls.append('body')
        runtime.set_forward_layout(batch)
        for layer in range(3, 3 + completed):
            runtime.complete_layer(layer)
        raise failure
    monkeypatch.setattr(pcp.GPUModelRunner, entry, fail)
    runner = pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    execute = getattr(runner, entry)
    with pytest.raises(RuntimeError) as error:
        execute(2)
    assert error.value is failure
    assert runtime._LAYOUT is None
    assert not runtime._FRAME_LOCK.locked()
    assert runtime._FRAME_POISONED
    with pytest.raises(RuntimeError, match='restart the engine'):
        execute(2)
    with pytest.raises(RuntimeError, match='restart the engine'):
        runtime.set_forward_layout(batch)
    assert calls == ['body']


def test_order_violation_clears_only_owning_frame(batch):
    runtime.set_forward_layout(batch)
    with pytest.raises(RuntimeError, match='order changed'):
        runtime.complete_layer(4)
    assert runtime._LAYOUT is None
    assert runtime._FRAME_POISONED
    assert not runtime._FRAME_LOCK.locked()


def test_foreign_thread_cannot_release_or_poison_active_owner(batch):
    runtime.set_forward_layout(batch)
    results = []
    def foreign():
        results.append(runtime.abort_model_forward())
        with pytest.raises(RuntimeError, match='order changed'):
            runtime.complete_layer(3)
    thread = threading.Thread(target=foreign)
    thread.start(); thread.join(3)
    assert not thread.is_alive()
    assert results == [False]
    assert runtime._FRAME_LOCK.locked() and not runtime._FRAME_POISONED
    runtime.complete_layer(3); runtime.complete_layer(4)
    assert not runtime._FRAME_LOCK.locked()


def test_successive_healthy_forwards_keep_existing_results(monkeypatch, batch):
    def forward(self, *args):
        runtime.set_forward_layout(batch)
        runtime.complete_layer(3); runtime.complete_layer(4)
        return 'result'
    monkeypatch.setattr(pcp.GPUModelRunner, 'execute_model', forward)
    runner = pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    assert runner.execute_model() == runner.execute_model() == 'result'
    assert not runtime._FRAME_LOCK.locked() and not runtime._FRAME_POISONED


def test_unrelated_failure_without_static_frame_is_not_latched(monkeypatch, batch):
    def fail(self): raise ValueError('before any static frame')
    monkeypatch.setattr(pcp.GPUModelRunner, 'execute_model', fail)
    runner = pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    with pytest.raises(ValueError): runner.execute_model()
    assert runtime._LAYOUT is None
    assert not runtime._FRAME_LOCK.locked() and not runtime._FRAME_POISONED


def test_failed_frame_setup_releases_lock_and_latches_restart(monkeypatch, batch):
    def fail_setup(batch):
        raise MemoryError('injected frame metadata allocation failure')
    monkeypatch.setattr(runtime, '_set_forward_layout', fail_setup)
    with pytest.raises(MemoryError): runtime.set_forward_layout(batch)
    assert runtime._LAYOUT is None and runtime._FRAME_POISONED
    assert not runtime._FRAME_LOCK.locked()
    with pytest.raises(RuntimeError, match='restart the engine'):
        runtime.set_forward_layout(batch)


def test_msa_layout_failure_is_cleaned_by_actual_runner_boundary(monkeypatch,batch):
    from hcu_vllm_das.parallel import minimax31_msa as msa
    manager=pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    manager._hcu_native_msa=True;manager._hcu_static=True
    monkeypatch.setattr(pcp.PCPManager,'partition_batch',lambda *a:batch)
    error=RuntimeError('MSA planner start/lens contract mismatch')
    def fail(batch): raise error
    monkeypatch.setattr(msa,'set_forward_layout',fail)
    monkeypatch.setattr(pcp.GPUModelRunner,'execute_model',lambda self:manager.partition_batch(None,None))
    runner=pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    with pytest.raises(RuntimeError) as caught:runner.execute_model()
    assert caught.value is error
    assert runtime._LAYOUT is None and not runtime._FRAME_LOCK.locked()
    assert runtime._FRAME_POISONED
    with pytest.raises(RuntimeError,match='restart the engine'):runner.execute_model()
