# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the PCP callback and global Engram group ordering."""
from datetime import timedelta
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import v41_engram_edp as edp


@pytest.mark.parametrize('rank', range(4))
def test_edp_builds_all_tp2_dp2_groups_in_world_order(monkeypatch, rank):
    partitions = [[0, 2], [1, 3]]
    dp = NS(world_size=2, ranks=partitions[rank % 2],
            group_ranks=partitions, device_group=object())
    calls = []
    dedicated = object()

    def initialize(group_ranks, local_rank, backend, *, group_name):
        calls.append((group_ranks, local_rank, backend, group_name))
        return dedicated

    state = NS(get_dp_group=lambda: dp, get_engram_dp_group=lambda: dp,
               get_world_group=lambda: NS(local_rank=rank),
               init_model_parallel_group=initialize)
    monkeypatch.setattr(edp, '_parallel_state', state)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda group: 'gloo')
    edp._restore_independent_edp()
    assert calls == [(partitions, rank, 'gloo', 'edp')]
    assert state._ENGRAM_DP is dedicated


@pytest.mark.parametrize('actual', [0, 1, 3])
def test_pcp_first_forward_resolves_context_without_touching_lazy_export(actual):
    # A fresh interpreter is essential: earlier patch.object(prod, ...) calls
    # materialize __getattr__ exports and used to hide the missing global.
    program = r'''
from types import SimpleNamespace as NS
import torch
import vllm.forward_context as context
from hcu_vllm_das import v41_pcp as prod
from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
actual = ACTUAL
calls = []
context.get_forward_context = lambda: NS(attn_metadata={'swa': NS(
    num_decode_tokens=actual, num_prefill_tokens=0)})
def native(self, q, kv, positions, output):
    calls.append(q.shape[0])
    output.copy_(q * 2)
DeepseekV41ROCMAiterMLAAttention.forward_mqa = native
assert 'get_forward_context' not in vars(prod)
cls = prod.attention_class()
obj = object.__new__(cls)
torch.nn.Module.__init__(obj)
obj.swa_cache_layer = NS(prefix='swa')
q = torch.ones(5, 2, 4)
output = torch.full_like(q, 777)
obj.forward_mqa(q, torch.ones(5, 4), torch.arange(5), output)
assert calls == ([] if actual == 0 else [actual]), calls
assert (output[:actual] == 2).all() and (output[actual:] == 0).all()
assert not torch.cuda.is_initialized()
'''.replace('ACTUAL', str(actual))
    result = subprocess.run([sys.executable, '-B', '-c', program],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def _edp_gloo_worker(rank, rendezvous):
    from vllm.distributed import parallel_state as native
    from vllm.distributed import utils

    timeout = timedelta(seconds=20)
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timeout)
    # Bound subgroup rendezvous as well as the world-group timeout.
    utils.get_distributed_timeout_or_none = lambda: timeout
    utils.get_cpu_distributed_timeout_or_none = lambda: timeout
    partitions = [[0, 2], [1, 3]]
    dp = native.GroupCoordinator(partitions, rank, 'gloo', False,
                                 group_name='review-dp')

    def initialize(group_ranks, local_rank, backend, *, group_name):
        # The real native constructor and torch.new_group execute. Only the
        # HCU device communicator is disabled in this CPU integration test.
        return native.init_model_parallel_group(
            group_ranks, local_rank, backend, group_name=group_name,
            use_device_communicator=False)

    state = NS(get_dp_group=lambda: dp, get_engram_dp_group=lambda: dp,
               get_world_group=lambda: NS(local_rank=rank),
               init_model_parallel_group=initialize)
    edp._parallel_state = state
    try:
        edp._restore_independent_edp()
        dedicated = state._ENGRAM_DP
        assert dedicated.group_ranks == partitions
        assert dedicated.ranks == partitions[rank % 2]
        assert dedicated.device_group is not dp.device_group
        value = torch.tensor([rank + 1.0])
        torch.distributed.all_reduce(value, group=dedicated.device_group)
        assert value.item() == (4.0 if rank % 2 == 0 else 6.0)
        torch.distributed.barrier()
        dedicated.destroy()
        assert not torch.cuda.is_initialized()
    finally:
        dp.destroy()
        torch.distributed.destroy_process_group()


def test_edp_real_four_process_gloo_groups(tmp_path):
    import torch.multiprocessing as mp

    context = mp.spawn(_edp_gloo_worker,
                       args=(str(tmp_path / 'rendezvous'),), nprocs=4, join=False)
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail('four-process Engram EDP rendezvous timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
