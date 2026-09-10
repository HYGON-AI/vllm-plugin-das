from inspect import signature

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2


def test_prepare_dummy_attn_matches_vllm_parameter_contract() -> None:
    upstream = signature(GPUModelRunner.prepare_dummy_attn).parameters
    hcu = signature(HcuGPUModelRunnerV2.prepare_dummy_attn).parameters

    assert tuple(hcu) == tuple(upstream)
    assert hcu["valid_state_slots"].default is False
