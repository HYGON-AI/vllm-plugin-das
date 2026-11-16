# SPDX-License-Identifier: Apache-2.0
"""Keep DP agreement reuse native and free of rank-local recovery collectives.

These are CPU branch/contract tests, not a real distributed execution test.
Only the graph manager and communication boundary are mocked.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class _GraphManager:
    def __init__(self, padding=0):
        self.padding = padding
        self.calls = []

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, **kwargs):
        self.calls.append((num_reqs, num_tokens, uniform_token_count))
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=num_tokens + self.padding,
            num_reqs=num_reqs,
        )


@pytest.fixture
def forbid_collective(monkeypatch):
    def unexpected_sync(*args, **kwargs):
        pytest.fail('reusing DP agreement must not start a rank-local collective')

    monkeypatch.setattr(dp_utils, 'sync_cudagraph_and_dp_padding', unexpected_sync)


def _sync(tokens, uniform_token_count, eager=False):
    return dp_utils.DPSyncState(
        num_tokens_across_dp=torch.tensor(tokens, dtype=torch.int32),
        uniform_token_count=uniform_token_count,
        eager=eager,
        num_reqs=2,
    )


def test_step5_does_not_register_global_dp_recovery():
    from vllm_hcu.patch import worker

    adapters = [spec.adapter for group in worker._ALL_CALLBACK_GROUPS for spec in group]
    assert not any(name.endswith('.patch_step5_dp_sync') for name in adapters)
    assert not hasattr(dp_utils, '_vllm_hcu_step5_dp_sync_applied')


@pytest.mark.parametrize('dp_rank', [0, 1])
@pytest.mark.parametrize('synced_uniform,local_uniform', [(4, 4), (None, 4), (None, None)])
def test_graph_agreement_reuse_needs_no_new_collective(
    forbid_collective, dp_rank, synced_uniform, local_uniform
):
    sync = _sync([8, 8], synced_uniform)
    manager = _GraphManager()
    desc, reused = dp_utils.dispatch_cg_and_sync_dp(
        manager, 2, 8, local_uniform, 2, dp_rank, dp_sync=sync
    )
    assert desc.num_tokens == 8 and reused is sync
    assert manager.calls == [(2, 8, synced_uniform)]


@pytest.mark.parametrize('dp_rank,num_tokens', [(0, 4), (1, 8)])
def test_eager_agreement_keeps_each_ranks_token_count(
    forbid_collective, dp_rank, num_tokens
):
    sync = _sync([4, 8], None, eager=True)
    desc, reused = dp_utils.dispatch_cg_and_sync_dp(
        None, 2, num_tokens, None, 2, dp_rank, dp_sync=sync
    )
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_tokens == num_tokens and reused is sync


@pytest.mark.parametrize('dp_rank', [0, 1])
def test_drafter_extra_padding_updates_whole_agreement_without_collective(
    forbid_collective, dp_rank
):
    sync = _sync([8, 8], 4)
    desc, reused = dp_utils.dispatch_cg_and_sync_dp(
        _GraphManager(padding=8), 2, 8, 4, 2, dp_rank, dp_sync=sync
    )
    assert desc.num_tokens == 16
    assert reused.num_tokens_across_dp.tolist() == [16, 16]
    assert sync.num_tokens_across_dp.tolist() == [8, 8]


def test_one_stale_rank_fails_contract_without_starting_new_collective(forbid_collective):
    sync = _sync([8, 8], 4)
    # Only rank 0 changed its batch. It must fail the caller contract instead
    # of entering a collective while rank 1 reuses the old agreement.
    with pytest.raises(AssertionError, match='different batch'):
        dp_utils.dispatch_cg_and_sync_dp(
            _GraphManager(), 1, 4, 4, 2, 0, dp_sync=sync
        )
    _, reused = dp_utils.dispatch_cg_and_sync_dp(
        _GraphManager(), 2, 8, 4, 2, 1, dp_sync=sync
    )
    assert reused is sync


@pytest.mark.parametrize('local_uniform', [None, 1])
def test_changed_uniform_length_keeps_native_fail_fast_contract(
    forbid_collective, local_uniform
):
    with pytest.raises(AssertionError, match='different batch'):
        dp_utils.dispatch_cg_and_sync_dp(
            _GraphManager(), 2, 8, local_uniform, 2, 0, dp_sync=_sync([8, 8], 4)
        )


def test_single_rank_dispatch_needs_no_collective(forbid_collective):
    desc, sync = dp_utils.dispatch_cg_and_sync_dp(_GraphManager(), 2, 8, 4, 1, 0)
    assert desc.num_tokens == 8 and sync is None


@pytest.mark.parametrize('dp_rank', [0, 1])
def test_fresh_agreement_still_uses_exactly_one_collective(monkeypatch, dp_rank):
    calls = []
    cpu_group = object()
    monkeypatch.setattr(dp_utils, 'get_dp_group', lambda: SimpleNamespace(cpu_group=cpu_group))

    def all_reduce(tensor, *, group):
        calls.append(group)
        assert group is cpu_group
        # Emulate the result of both ranks contributing their current batch.
        tensor.copy_(torch.tensor([[8, 4], [0, 0], [4, 4], [4, 4], [0, 0], [2, 1]]))

    monkeypatch.setattr(dp_utils.dist, 'all_reduce', all_reduce)
    num_tokens = [8, 4][dp_rank]
    desc, sync = dp_utils.dispatch_cg_and_sync_dp(
        None, [2, 1][dp_rank], num_tokens, 4, 2, dp_rank,
        need_eager=True, max_query_len=4,
    )
    assert len(calls) == 1
    assert desc.num_tokens == num_tokens and sync.eager
    assert sync.num_tokens_across_dp.tolist() == [8, 4]
    assert sync.uniform_token_count == 4
