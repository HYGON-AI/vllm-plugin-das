from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.v1.hcu_model_runner_v2 import (
    HcuGPUModelRunnerV2,
    sampler_warmup_scope,
    suppress_sampler_warmup_logprobs,
)


def test_runner_v2_overrides_match_installed_vllm_parameter_contract() -> None:
    # Runner V2 owns different call contracts from the legacy HCU runner.
    # Keep every override that crosses that boundary aligned with the actual
    # installed vLLM API, including HCU's allocation-context arguments.
    methods = (
        "initialize_kv_cache",
        "prepare_inputs",
        "prepare_attn",
        "prepare_dummy_attn",
        "execute_model",
        "sample_tokens",
    )
    for name in methods:
        upstream = signature(getattr(GPUModelRunner, name)).parameters
        hcu = signature(getattr(HcuGPUModelRunnerV2, name)).parameters
        assert tuple(hcu) == tuple(upstream), name

    dummy_attn = signature(HcuGPUModelRunnerV2.prepare_dummy_attn).parameters
    assert dummy_attn["valid_state_slots"].default is False


def test_sampler_warmup_logprobs_are_suppressed_and_restored() -> None:
    num_logprobs = np.array([5, 3, -1, 7], dtype=np.int32)
    model_runner = SimpleNamespace(
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(num_logprobs=num_logprobs)
        )
    )
    warmup_batch = SimpleNamespace(
        req_ids=["_warmup_0_", "_warmup_1_"],
        num_reqs=2,
        idx_mapping_np=np.array([3, 1], dtype=np.intp),
    )

    with pytest.raises(RuntimeError, match="synthetic warmup"):
        with sampler_warmup_scope():
            with suppress_sampler_warmup_logprobs(model_runner, warmup_batch):
                assert num_logprobs.tolist() == [5, -1, -1, -1]
                raise RuntimeError("synthetic warmup")

    assert num_logprobs.tolist() == [5, 3, -1, 7]


@pytest.mark.parametrize("request_id", ["request-0", "_warmup_customer-job"])
def test_sampler_logprobs_are_unchanged_for_real_requests(request_id) -> None:
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    num_logprobs = np.array([5, 3], dtype=np.int32)
    states = SamplingStates.__new__(SamplingStates)
    states.num_logprobs = num_logprobs
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = states
    model_runner = SimpleNamespace(
        sampler=sampler
    )
    real_batch = SimpleNamespace(
        req_ids=[request_id],
        num_reqs=1,
        idx_mapping_np=np.array([0], dtype=np.intp),
    )

    with suppress_sampler_warmup_logprobs(model_runner, real_batch):
        assert num_logprobs.tolist() == [5, 3]
        assert sampler.get_logprobs_dims(real_batch.idx_mapping_np, False) == (5, 0)

    assert num_logprobs.tolist() == [5, 3]


def test_worker_startup_controls_sampler_scope_and_restores_on_error(monkeypatch):
    from vllm.v1.worker.gpu_worker import Worker
    from vllm_hcu.v1.worker import HcuGPUWorker

    values = np.array([5], dtype=np.int32)
    runner = SimpleNamespace(
        sampler=SimpleNamespace(sampling_states=SimpleNamespace(num_logprobs=values))
    )
    batch = SimpleNamespace(num_reqs=1, idx_mapping_np=np.array([0]))

    def startup(self):
        with suppress_sampler_warmup_logprobs(self.model_runner, batch):
            assert values.tolist() == [-1]
        raise RuntimeError("startup failed")

    monkeypatch.setattr(Worker, "compile_or_warm_up_model", startup)
    worker = HcuGPUWorker.__new__(HcuGPUWorker)
    worker.use_v2_model_runner = True
    worker.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    worker.model_runner = runner
    with pytest.raises(RuntimeError, match="startup failed"):
        worker.compile_or_warm_up_model()
    with suppress_sampler_warmup_logprobs(runner, batch):
        assert values.tolist() == [5]
