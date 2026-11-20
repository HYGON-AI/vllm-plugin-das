# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Configuration policy and forward-local metadata ownership; no GPU required."""

from hcu_vllm_das.adapter_api import adapter_module as _adapter_module
import json
import os
from types import SimpleNamespace as NS
import pytest
import torch
hcu_config = _adapter_module('hcu_config')
registry = _adapter_module('registry')
from hcu_vllm_das.parallel import minimax31_prefill as opt
from hcu_vllm_das.parallel import minimax31_msa as msa

OP = 'vllm_minimax_prefill'

@pytest.fixture
def policy(monkeypatch, tmp_path):
    config = tmp_path/'hcu_adapter.json'
    config.write_text(json.dumps({'version': 1, 'ops': {}}))
    monkeypatch.setenv('HCU_CONFIG', str(config))
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': False, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(opt, '_POLICY', None)
    monkeypatch.setattr(opt, '_ENABLED', None)
    monkeypatch.setattr(opt, '_PID', os.getpid())
    monkeypatch.setattr(opt, '_PACK', {})
    monkeypatch.setattr(opt, '_ACTIVE', False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    return config


def test_default_is_optimized_without_private_environment(policy, monkeypatch):
    monkeypatch.setenv('HCU_MINIMAX_PREFILL_OPT', 'off')  # obsolete input has no owner
    assert opt.enabled()
    assert opt._POLICY == ('triton', False)


@pytest.mark.parametrize('kind', ['config', 'veto'])
def test_stock_configuration_and_startup_veto_preserve_comparison(policy, monkeypatch, kind):
    if kind == 'config':
        policy.write_text(json.dumps({'version': 1, 'ops': {OP: 'stock'}}))
    else:
        policy.write_text(json.dumps({'version': 1, 'ops': {OP: 'triton'}}))
        monkeypatch.setenv('HCU_DISABLE_OPS', OP)
    assert not opt.enabled()


def test_policy_changes_after_use_are_rejected(policy):
    assert opt.enabled()
    policy.write_text(json.dumps({'version': 1, 'ops': {OP: 'stock'}}))
    hcu_config.load(force=True)
    with pytest.raises(RuntimeError, match='restart'):
        opt.validate_policy()


def test_active_fork_is_rejected_before_inherited_state(policy):
    if not hasattr(os, 'fork'): pytest.skip('requires fork')
    import select, signal
    assert opt.enabled()
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            opt.enabled()
            result = b'accepted'
        except RuntimeError as error:
            result = str(error).encode()
        os.write(write, result); os._exit(0)
    os.close(write)
    try:
        assert select.select([read], [], [], 3)[0]
        assert b'inherited across fork' in os.read(read, 2048)
    finally:
        os.close(read)
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
        os.waitpid(pid, 0)


def test_metadata_reuse_tracks_physical_storage_versions_and_forward(monkeypatch):
    monkeypatch.setattr(msa, '_prefill_enabled', lambda: True)
    table = torch.arange(16, dtype=torch.int32).view(2, 8)
    seq = torch.tensor([132, 259], dtype=torch.int32)
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)
    batch = NS(num_scheduled_tokens=torch.tensor([4, 3]),
               num_computed_tokens_np=torch.tensor([128, 256]), num_tokens=7)
    msa.set_forward_layout(batch)
    a = NS(block_table=table, seq_lens=seq, cu_seqlens_q=cu)
    first = msa._forward_meta(a, 0, 2)
    assert msa._forward_meta(NS(block_table=table, seq_lens=seq, cu_seqlens_q=cu), 0, 2) is first
    table.add_(7)
    changed = msa._forward_meta(a, 0, 2)
    assert changed is not first
    assert not torch.equal(first.packed_kv_pages, changed.packed_kv_pages)
    msa.set_forward_layout(batch)
    assert msa._forward_meta(a, 0, 2) is not changed
    seq.add_(1)
    with pytest.raises(RuntimeError, match='seq_lens mismatch'):
        msa._forward_meta(a, 0, 2)


def test_rejected_static_frame_cannot_overwrite_msa_metadata(monkeypatch):
    from hcu_vllm_das.parallel import minimax31_pcp as pcp
    from hcu_vllm_das.parallel import minimax31_static_runtime as static
    writes = []
    monkeypatch.setattr(pcp.PCPManager, 'partition_batch', lambda *a: object())
    monkeypatch.setattr(msa, 'set_forward_layout', lambda batch: writes.append(batch))
    def reject(batch): raise RuntimeError('another forward owns the static frame')
    monkeypatch.setattr(static, 'set_forward_layout', reject)
    manager = pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    manager._hcu_native_msa = True
    manager._hcu_static = True
    with pytest.raises(RuntimeError, match='another forward'):
        manager.partition_batch(None, None)
    assert writes == []


def test_metadata_holds_tensor_sources_when_wrapper_fields_change(monkeypatch):
    import gc, weakref
    monkeypatch.setattr(msa, '_prefill_enabled', lambda: True)
    batch = NS(num_scheduled_tokens=torch.tensor([2]),
               num_computed_tokens_np=torch.tensor([0]), num_tokens=2)
    msa.set_forward_layout(batch)
    table = torch.arange(4, dtype=torch.int32).view(1, 4)
    ref = weakref.ref(table)
    sub = NS(block_table=table, seq_lens=torch.tensor([2], dtype=torch.int32),
             cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32))
    msa._forward_meta(sub, 0, 1)
    sub.block_table = table.clone()
    del table
    gc.collect()
    assert ref() is not None, 'cached addresses must retain tensors, not a mutable wrapper'
    msa.set_forward_layout(batch)
    gc.collect()
    assert ref() is None


def test_initialized_layer_policy_does_not_reload_configuration(policy, monkeypatch):
    assert opt.enabled()
    def no_read(*args, **kwargs): raise AssertionError('per-layer policy read')
    monkeypatch.setattr(hcu_config,'load',no_read)
    monkeypatch.setattr(opt,'op_disabled',no_read)
    assert opt.enabled() and opt.enabled()


def test_planner_boundary_rejects_startup_veto_drift(policy, monkeypatch):
    assert opt.enabled()
    monkeypatch.setenv('HCU_DISABLE_OPS',OP)
    with pytest.raises(RuntimeError,match='restart'):
        opt.validate_policy()


def test_paired_adapter_feature_floor_is_explicit(policy, monkeypatch):
    monkeypatch.delitem(hcu_config.BACKEND_CATALOG,OP)
    with pytest.raises(RuntimeError,match='paired Adapter policy catalog'):
        opt.enabled()


@pytest.mark.parametrize('optimized',[False,True])
def test_metadata_retains_only_source_tensors_not_the_wrapper(monkeypatch,optimized):
    import gc,weakref
    class Metadata: pass
    monkeypatch.setattr(msa,'_prefill_enabled',lambda:optimized)
    batch=NS(num_scheduled_tokens=torch.tensor([2]),num_computed_tokens_np=torch.tensor([0]),num_tokens=2)
    msa.set_forward_layout(batch)
    sub=Metadata()
    sub.block_table=torch.arange(4,dtype=torch.int32).view(1,4)
    sub.seq_lens=torch.tensor([2],dtype=torch.int32)
    sub.cu_seqlens_q=torch.tensor([0,2],dtype=torch.int32)
    sub.unrelated=torch.zeros(128)
    wrapper=weakref.ref(sub);unrelated=weakref.ref(sub.unrelated);table=weakref.ref(sub.block_table)
    msa._forward_meta(sub,0,1)
    del sub;gc.collect()
    assert wrapper() is None and unrelated() is None
    assert table() is not None
    msa.set_forward_layout(batch);gc.collect()
    assert table() is None


def test_planner_rejects_policy_drift_before_mutating_inputs(policy,monkeypatch):
    from hcu_vllm_das.parallel import minimax31_pcp as pcp
    calls=[]
    assert opt.enabled()
    monkeypatch.setenv('HCU_DISABLE_OPS',OP)
    monkeypatch.setattr(pcp.PCPManager,'partition_batch',lambda *a:calls.append('planner'))
    manager=pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    with pytest.raises(RuntimeError,match='restart'):manager.partition_batch(None,None)
    assert calls==[]


@pytest.mark.parametrize('global_rows',[65,2048])
def test_existing_runner_memory_probe_reserves_the_local_pack_budget(monkeypatch,global_rows):
    from hcu_vllm_das.parallel import minimax31_pcp as pcp
    from hcu_vllm_das.parallel import minimax31_static_runtime as runtime
    monkeypatch.setattr(runtime,'_FRAME_POISONED',False)
    monkeypatch.setattr(pcp,'_validate_prefill_policy',lambda:True)
    monkeypatch.setattr(pcp,'_prefill_enabled',lambda:True)
    monkeypatch.setattr(pcp.PCPManager,'prepare_inputs_for_capture_or_profile',lambda self,b:b)
    reserved=[];probed=[]
    monkeypatch.setattr(pcp,'reserve_pack_rows',lambda rows,device:reserved.append((rows,device)))
    manager=pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    manager._hcu_native_msa=True;manager._hcu_static=False
    manager.pcp_world_size=8;manager.dcp_world_size=1;manager.shard_decode_requests=True
    def dummy(self,num_tokens,*a,**kw):
        assert kw['is_profile'] and kw['skip_attn']
        probed.append(num_tokens)
        return manager.prepare_inputs_for_capture_or_profile(NS(num_tokens_after_padding=num_tokens,positions=torch.arange(num_tokens)))
    monkeypatch.setattr(pcp.GPUModelRunner,'_dummy_run',dummy)
    runner=pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    runner.pcp_manager=manager;runner.max_num_reqs=8;runner.speculator=None;runner.device=torch.device('cpu')
    runner._dummy_run(global_rows,is_profile=True,skip_attn=True)
    assert len(probed)==1 and 0<probed[0]<=global_rows
    assert reserved==[(probed[0],torch.device('cpu'))]


def test_non_profile_dummy_preparation_cannot_grow_pack_scratch(monkeypatch):
    from hcu_vllm_das.parallel import minimax31_pcp as pcp
    monkeypatch.setattr(pcp,'_validate_prefill_policy',lambda:True)
    monkeypatch.setattr(pcp,'_prefill_enabled',lambda:True)
    monkeypatch.setattr(pcp.PCPManager,'prepare_inputs_for_capture_or_profile',lambda self,b:b)
    def no_allocation(*a):raise AssertionError('scratch grew outside the memory probe')
    monkeypatch.setattr(pcp,'reserve_pack_rows',no_allocation)
    manager=pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    manager._hcu_native_msa=True;manager._hcu_static=False
    batch=NS(num_tokens_after_padding=4096,positions=torch.arange(4096))
    assert manager.prepare_inputs_for_capture_or_profile(batch) is batch
