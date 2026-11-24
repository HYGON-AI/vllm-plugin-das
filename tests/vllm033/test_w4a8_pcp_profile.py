from collections import namedtuple
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from hcu_vllm_das.dsv41 import w4a8_profile as profile
from hcu_vllm_das.dsv41 import profile_worker


@dataclass
class Batch:
    input_ids: torch.Tensor
    num_scheduled_tokens: np.ndarray
    query_start_loc_np: np.ndarray
    is_padding: torch.Tensor

    @property
    def num_tokens(self):
        return int(self.query_start_loc_np[-1])

    @property
    def num_tokens_after_padding(self):
        return self.input_ids.numel()


def make_batch(counts, padded=0):
    offsets = np.concatenate(([0], np.cumsum(counts)))
    ids = torch.zeros(int(offsets[-1]) + padded, dtype=torch.int32)
    return Batch(ids, np.asarray(counts), offsets, torch.zeros_like(ids, dtype=torch.bool))


def make_manager(rank, ranks=8):
    manager = object.__new__(profile.W4A8ProfilePCPManager)
    manager.pcp_rank, manager.pcp_world_size, manager.dcp_world_size = rank, ranks, 1
    manager.shard_decode_requests = False
    return manager


@pytest.mark.parametrize("counts,padded", [([256] * 128, 0), ([1, 19, 221, 301], 31)])
@pytest.mark.parametrize("ranks", [4, 8])
def test_profile_assigns_each_prefill_row_once_using_core_pcp(counts, padded, ranks):
    batch = make_batch(counts, padded)
    ownership = []
    for rank in range(ranks):
        mask, owned = profile.target_profile_padding(make_manager(rank, ranks), batch)
        ownership.append(~mask)
        if len(counts) == 128:
            assert owned == 32768 // ranks
        assert not batch.input_ids.any()
    totals = torch.stack(ownership).sum(0)
    torch.testing.assert_close(totals[:batch.num_tokens], torch.ones_like(totals[:batch.num_tokens]))
    assert not totals[batch.num_tokens:].any()


def test_workspaces_cover_top6_and_replicated_decode_with_eplb():
    routed = profile.routed_token_bound(32768, 128, 6, 8, False)
    assert routed == 32768 + 7 * 128 * 6
    assert profile.routed_token_bound(32768, 128, 6, 8, True) == 32768
    reserved = profile.contiguous_workspace_bytes(routed, 6, 52, 2304, 5120)
    assert 4 * 2**30 < reserved < 5 * 2**30
    assignments = np.zeros(52, dtype=np.int64)
    assignments[:51] = 257
    assignments[-1] = routed * 6 - assignments.sum()
    actual_rows = (((assignments + 255) // 256) * 256).sum()
    assert reserved >= actual_rows * 2 * 5120 * 2


@pytest.mark.parametrize("ranks", [4, 8])
def test_selects_only_native_eager_w4a8_with_padding_support(monkeypatch, ranks):
    monkeypatch.setattr(profile.envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(profile.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.delenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", raising=False)
    parallel = NS(tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
                  prefill_context_parallel_size=ranks, decode_context_parallel_size=1,
                  enable_expert_parallel=True, all2all_backend="deepep_high_throughput")
    config = NS(
        parallel_config=parallel, kernel_config=NS(moe_backend="deep_gemm"),
        model_config=NS(hf_config=NS(model_type="deepseek_v41"),
                        enforce_eager=True, quantization=profile.NAME),
    )
    assert profile_worker.w4a8_profile_selected(config)
    monkeypatch.setattr(profile.envs, "VLLM_MOE_SKIP_PADDING", False)
    assert not profile_worker.w4a8_profile_selected(config)


@pytest.mark.parametrize("layer_split_available", [False, True])
def test_worker_uses_prefill_wrapper_when_available(monkeypatch, layer_split_available):
    module_name = "hcu_vllm_das.dsa_prefill_runner"
    companion = None
    calls = []
    if layer_split_available:
        companion = ModuleType(module_name)

        def wrap(base, config, device):
            calls.append(base)
            return base(config, device)

        companion.wrap_profile_runner = wrap
    monkeypatch.setitem(sys.modules, module_name, companion)
    worker = object.__new__(profile.W4A8PcpProfileWorker)
    worker.vllm_config = object()
    worker.device = torch.device("cpu")
    expected = (worker.vllm_config, worker.device)
    monkeypatch.setattr(profile, "W4A8PcpProfileRunner", lambda config, device: (config, device))
    assert worker._make_model_runner() == expected
    assert calls == ([profile.W4A8PcpProfileRunner] if layer_split_available else [])


def test_reserves_workspace_before_core_measures_forward(monkeypatch):
    manager = make_manager(0)
    runner = object.__new__(profile.W4A8PcpProfileRunner)
    runner.pcp_manager = manager
    runner.max_num_tokens, runner.max_num_reqs, runner.decode_query_len = 32768, 128, 6
    class Experts:
        pass

    experts = Experts()
    experts.quant_method = object.__new__(profile.HcuCTW4A8MoEMethod)
    experts.top_k, experts.local_num_experts = 6, 52
    experts.moe_config = NS(intermediate_size_per_partition=2304, hidden_dim=5120)
    runner.model = NS(modules=lambda: [experts])
    events = []
    monkeypatch.setattr(profile, "RoutedExperts", Experts)
    monkeypatch.setattr(profile, "current_workspace_manager",
                        lambda: NS(get_simultaneous=lambda *args: events.append(args)))
    monkeypatch.setattr(profile.GPUModelRunner, "profile_run",
                        lambda self, randomize_inputs: events.append(randomize_inputs))
    runner.profile_run(randomize_inputs=True)
    expected_bytes = profile.contiguous_workspace_bytes(38144, 6, 52, 2304, 5120)
    assert events == [(((expected_bytes,), torch.uint8),), False]


@pytest.mark.parametrize("fails", [False, True])
def test_scopes_moe_mask_to_target_and_restores_draft_batch(monkeypatch, fails):
    manager = make_manager(0)
    runner = object.__new__(profile.W4A8PcpProfileRunner)
    runner.pcp_manager = manager
    batch = make_batch([32])
    expected_mask, _ = profile.target_profile_padding(manager, batch)
    State = namedtuple("State", "input_batch hidden_states aux_hidden_states")
    runner.speculator = object()
    draft = torch.ones(32, 3)
    runner.model = NS(get_mtp_target_hidden_states=lambda: draft)
    monkeypatch.setattr(profile.PCPManager, "prepare_inputs_for_capture_or_profile",
                        lambda self, batch: batch)
    reductions = []

    def all_reduce(tensor):
        assert not tensor[expected_mask].any()
        reductions.append(tensor.shape)
        return torch.full_like(tensor, 2)

    monkeypatch.setattr(profile, "get_pcp_group", lambda: NS(all_reduce=all_reduce))

    def execute(self, scheduler_output, **kwargs):
        scoped = manager.prepare_inputs_for_capture_or_profile(batch)
        torch.testing.assert_close(scoped.is_padding, expected_mask)
        if fails:
            raise RuntimeError("profile failed")
        self.execute_model_state = State(scoped, torch.ones(32, 2), [torch.ones(32, 1)])

    monkeypatch.setattr(profile.GPUModelRunner, "execute_model", execute)
    flags = dict(is_profile=True, dummy_run=True, skip_attn_for_dummy_run=True)
    if fails:
        with pytest.raises(RuntimeError, match="profile failed"):
            runner.execute_model(None, **flags)
        assert reductions == []
    else:
        runner.execute_model(None, **flags)
        assert runner.execute_model_state.input_batch is batch
        assert runner.execute_model_state.hidden_states.eq(2).all()
        assert draft.eq(2).all()
        assert reductions == [(32, 3), (32, 2), (32, 1)]
    assert not manager.profile_target and manager.profile_batch is None
    assert manager.prepare_inputs_for_capture_or_profile(batch) is batch
