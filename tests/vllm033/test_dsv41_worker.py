"""Dummy-input policy only; Core retains profiling and PCP execution."""
from types import SimpleNamespace as NS
import pytest
import vllm.envs as envs
from hcu_vllm_das.dsv41.worker import HcuDSV41PcpWorker
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
    worker = object.__new__(HcuDSV41PcpWorker)
    worker.parallel_config = NS(prefill_context_parallel_size=pcp, data_parallel_size=dp)
    worker.vllm_config = NS(parallel_config=worker.parallel_config,
        kernel_config=NS(moe_backend=backend), model_config=NS(hf_config=NS(model_type=model)))
    assert worker.randomize_dummy_inputs is expected
    assert HcuDSV41PcpWorker.execute_model is Worker.execute_model


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("backend", ["deep_gemm_mega_moe", "deep_gemm"])
def test_memory_profile_uses_local_bound_and_restores_global_budget(monkeypatch, fail, backend):
    worker = object.__new__(HcuDSV41PcpWorker)
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
    from hcu_vllm_das.dsv41.profile_worker import local_token_bound
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
                    bound = local_token_bound(budget, max_reqs, 8)
                    assert actual <= bound, (lens, is_prefill, actual, bound)


def _profile_config(tmp_path, pcp=8, backend="deep_gemm_mega_moe", model="deepseek_v41"):
    return NS(parallel_config=NS(prefill_context_parallel_size=pcp, worker_cls="auto"),
              kernel_config=NS(moe_backend=backend),
              model_config=NS(hf_config=NS(model_type=model)))


@pytest.fixture
def platform(monkeypatch):
    """A throwaway Platform subclass so the real one is never patched."""
    from vllm.platforms.interface import Platform
    from hcu_vllm_das.dsv41 import profile_worker

    class Host(Platform):
        seen = []

        @classmethod
        def check_and_update_config(cls, vllm_config):
            # Mirrors Core (auto -> official) then HcuPlatform (official ->
            # Adapter) so an explicit operator choice survives both steps.
            parallel = vllm_config.parallel_config
            cls.seen.append(parallel.worker_cls)
            if parallel.worker_cls == "auto":
                parallel.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            if parallel.worker_cls == "vllm.v1.worker.gpu_worker.Worker":
                parallel.worker_cls = "hcu_adapter.bridges.vllm_worker.Worker"

    Host.device_name = "hcu"
    monkeypatch.setattr(profile_worker, "_installed", {})
    monkeypatch.delenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", raising=False)
    return Host, profile_worker


@pytest.mark.parametrize("enabled,pcp,backend,model,expected", [
    (True, 8, "deep_gemm_mega_moe", "deepseek_v41", "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"),
    (True, 8, "deep_gemm", "deepseek_v41", "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"),
    (False, 8, "deep_gemm_mega_moe", "deepseek_v41", "hcu_adapter.bridges.vllm_worker.Worker"),
    (None, 8, "deep_gemm_mega_moe", "deepseek_v41", "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"),
    (None, 8, "deep_gemm", "deepseek_v41", "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"),
    (None, 1, "deep_gemm_mega_moe", "deepseek_v41", "hcu_adapter.bridges.vllm_worker.Worker"),
    (None, 8, "triton", "deepseek_v41", "hcu_adapter.bridges.vllm_worker.Worker"),
    (None, 8, "deep_gemm_mega_moe", "other", "hcu_adapter.bridges.vllm_worker.Worker"),
    (True, 1, "deep_gemm_mega_moe", "deepseek_v41", "hcu_adapter.bridges.vllm_worker.Worker"),
    (True, 8, "triton", "deepseek_v41", "hcu_adapter.bridges.vllm_worker.Worker"),
    (True, 8, "deep_gemm_mega_moe", "other", "hcu_adapter.bridges.vllm_worker.Worker"),
])
def test_switch_selects_worker_only_for_pcp_channel_fp8(platform, monkeypatch,
                                                       enabled, pcp, backend, model, expected):
    if enabled is not None:
        monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "1" if enabled else "0")
    host, module = platform
    assert module.install(host)
    config = _profile_config(None, pcp, backend, model)
    host.check_and_update_config(config)
    assert config.parallel_config.worker_cls == expected
    # Core's own hook still ran first and its default was what we replaced.
    assert host.seen == ["auto"]


def test_explicit_worker_choice_is_never_overridden(platform, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "1")
    host, module = platform
    module.install(host)
    config = _profile_config(None)
    config.parallel_config.worker_cls = "user.module.CustomWorker"
    host.check_and_update_config(config)
    assert config.parallel_config.worker_cls == "user.module.CustomWorker"


def test_install_is_idempotent_and_keeps_core_behavior(platform):
    host, module = platform
    core = host.check_and_update_config.__func__
    assert module.install(host)
    wrapped = host.check_and_update_config.__func__
    assert module.install(host)
    assert host.check_and_update_config.__func__ is wrapped
    assert wrapped.__wrapped__ is core


def test_switch_defaults_on_and_rejects_non_boolean(platform, monkeypatch):
    from hcu_vllm_das.environment import dsv41_profile_worker_enabled

    assert dsv41_profile_worker_enabled() is True
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "maybe")
    with pytest.raises(ValueError, match="must be a boolean"):
        dsv41_profile_worker_enabled()


def test_auto_worker_keeps_adapter_lifecycle(monkeypatch):
    from hcu_adapter.bridges.vllm import HcuPlatform
    from hcu_adapter.bridges.vllm_worker import Worker as AdapterWorker
    from hcu_vllm_das.dsv41 import profile_worker
    from vllm.config.compilation import CUDAGraphMode
    from vllm.utils.import_utils import resolve_obj_by_qualname

    class Host(HcuPlatform):
        pass

    config = _profile_config(None)
    config.parallel_config.distributed_executor_backend = "mp"
    config.compilation_config = NS(cudagraph_mode=CUDAGraphMode.NONE)
    config.model_config.is_mm_prefix_lm = False
    config.scheduler_config = NS()
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_PROFILE_WORKER", "1")
    profile_worker.install(Host)
    Host.check_and_update_config(config)
    worker = resolve_obj_by_qualname(config.parallel_config.worker_cls)
    assert worker is HcuDSV41PcpWorker
    assert issubclass(worker, AdapterWorker)
    assert worker.load_model is AdapterWorker.load_model
    assert worker.reload_weights is AdapterWorker.reload_weights
