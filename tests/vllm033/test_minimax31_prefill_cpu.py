# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Configuration policy and forward-local metadata ownership; no GPU required."""
import json
import os
from types import SimpleNamespace as NS
import pytest
import torch
from hcu_adapter import hcu_config, registry
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
    monkeypatch.setattr(opt, '_PID', os.getpid())
    monkeypatch.setattr(opt, '_PACK', {})
    monkeypatch.setattr(opt, '_ACTIVE', False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    return config


def test_default_is_optimized_without_private_environment(policy, monkeypatch):
    monkeypatch.setenv('HCU_MINIMAX_PREFILL_OPT', 'off')  # obsolete input has no owner
    assert all(opt.enabled(name) for name in opt._FEATURES)
    assert opt._POLICY == ('triton', False)
    with pytest.raises(ValueError, match='unknown'):
        opt.enabled('misspelled')


@pytest.mark.parametrize('kind', ['config', 'veto'])
def test_stock_configuration_and_startup_veto_preserve_comparison(policy, monkeypatch, kind):
    if kind == 'config':
        policy.write_text(json.dumps({'version': 1, 'ops': {OP: 'stock'}}))
    else:
        monkeypatch.setenv('HCU_DISABLE_OPS', OP)
    assert not any(opt.enabled(name) for name in opt._FEATURES)


def test_policy_changes_after_use_are_rejected(policy):
    assert opt.enabled('q_fp8')
    policy.write_text(json.dumps({'version': 1, 'ops': {OP: 'stock'}}))
    hcu_config.load(force=True)
    with pytest.raises(RuntimeError, match='restart'):
        opt.enabled('q_fp8')


def test_active_fork_is_rejected_before_inherited_state(policy):
    if not hasattr(os, 'fork'): pytest.skip('requires fork')
    import select, signal
    assert opt.enabled('metadata')
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            opt.enabled('metadata')
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
    monkeypatch.setattr(msa, '_prefill_enabled', lambda n: n == 'metadata')
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
    monkeypatch.setattr(msa, '_prefill_enabled', lambda n: n == 'metadata')
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
