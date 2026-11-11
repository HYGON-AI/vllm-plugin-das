"""Dummy-input policy only; Core retains profiling and PCP execution."""
from types import SimpleNamespace as NS
import pytest
import vllm.envs as envs
from hcu_vllm_das.v41_worker import HcuV41PcpWorker
from vllm.v1.worker.gpu_worker import Worker


@pytest.mark.parametrize("pcp,dp,backend,model,flag,expected", [
    (8, 1, "deep_gemm_mega_moe", "deepseek_v41", False, True),
    (8, 1, "deep_gemm_mega_moe", "deepseek_v41", True, True),
    (8, 1, "deep_gemm", "deepseek_v41", False, True),
    (1, 1, "deep_gemm_mega_moe", "deepseek_v41", False, False),
    (8, 1, "triton", "deepseek_v41", False, False),
    (8, 1, "deep_gemm_mega_moe", "other", False, False),
    (1, 2, "triton", "other", True, True),
    (1, 2, "triton", "other", False, False),
])
def test_dummy_policy_is_scoped_and_preserves_core(monkeypatch, pcp, dp, backend, model, flag, expected):
    monkeypatch.setattr(envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", flag)
    worker = object.__new__(HcuV41PcpWorker)
    worker.parallel_config = NS(prefill_context_parallel_size=pcp, data_parallel_size=dp)
    worker.vllm_config = NS(parallel_config=worker.parallel_config,
        kernel_config=NS(moe_backend=backend), model_config=NS(hf_config=NS(model_type=model)))
    assert worker.randomize_dummy_inputs is expected
    assert HcuV41PcpWorker.execute_model is Worker.execute_model


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("backend", ["deep_gemm_mega_moe", "deep_gemm"])
def test_memory_profile_uses_local_bound_and_restores_global_budget(monkeypatch, fail, backend):
    worker = object.__new__(HcuV41PcpWorker)
    worker.rank = 0
    worker.parallel_config = NS(prefill_context_parallel_size=8, data_parallel_size=1)
    worker.vllm_config = NS(parallel_config=worker.parallel_config,
        kernel_config=NS(moe_backend=backend),
        model_config=NS(hf_config=NS(model_type="deepseek_v41")),
        scheduler_config=NS(max_num_batched_tokens=32768))
    worker.model_runner = NS(max_num_tokens=32768, max_num_reqs=128, pcp_manager=object())
    def core_profile(self):
        assert self.model_runner.max_num_tokens == 4352
        assert self.vllm_config.scheduler_config.max_num_batched_tokens == 32768
        if fail:
            raise RuntimeError("profile failed")
        return 123456
    monkeypatch.setattr(Worker, "determine_available_memory", core_profile)
    if fail:
        with pytest.raises(RuntimeError, match="profile failed"):
            worker.determine_available_memory()
    else:
        assert worker.determine_available_memory() == 123456
    assert worker.model_runner.max_num_tokens == 32768


def test_profile_bound_covers_core_pcp_mixed_and_rounded_batches():
    import numpy as np
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from hcu_vllm_das.v41_worker import _local_profile_token_bound
    rng = np.random.default_rng(42)
    manager = object.__new__(PCPManager)
    manager.pcp_world_size, manager.dcp_world_size = 8, 1
    for shard_decodes in (False, True):
        manager.shard_decode_requests = shard_decodes
        for budget in (7, 2048, 32768):
            for max_reqs in (64, 128):
                for _ in range(30):
                    reqs = min(max_reqs, budget)
                    is_prefill = rng.integers(0, 2, reqs).astype(bool)
                    is_prefill[0] = True
                    lens = np.ones(reqs, dtype=np.int32)
                    remaining = budget - reqs
                    lens[is_prefill] += rng.multinomial(remaining,
                        np.full(is_prefill.sum(), 1 / is_prefill.sum()))
                    actual = manager.get_num_tokens_for_dispatch(lens, is_prefill)
                    bound = _local_profile_token_bound(budget, max_reqs, 8)
                    assert actual <= bound, (lens, is_prefill, actual, bound)
