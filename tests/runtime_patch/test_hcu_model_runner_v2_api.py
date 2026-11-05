from inspect import signature
from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.v1.hcu_model_runner_v2 import (
    HcuGPUModelRunnerV2,
    _dsv4_pcp_experimental_enabled,
)


def test_prepare_dummy_attn_matches_vllm_parameter_contract() -> None:
    upstream = signature(GPUModelRunner.prepare_dummy_attn).parameters
    hcu = signature(HcuGPUModelRunnerV2.prepare_dummy_attn).parameters

    assert tuple(hcu) == tuple(upstream)
    assert hcu["valid_state_slots"].default is False


@pytest.mark.parametrize(
    ("architecture", "experimental", "expected"),
    [
        ("Qwen3ForCausalLM", True, False),
        ("DeepseekV41ForCausalLM", False, False),
        ("DeepseekV41ForCausalLM", True, True),
        ("DeepseekV4ForCausalLM", True, True),
    ],
)
def test_dsv4_experimental_gate_controls_multi_group_pcp(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    experimental: bool,
    expected: bool,
) -> None:
    """Only explicitly armed DSV4 PCP may use hybrid KV groups."""

    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
    )
    if experimental:
        monkeypatch.setenv("VLLM_HCU_DSV4_PCP_EXPERIMENTAL", "1")
    else:
        monkeypatch.delenv("VLLM_HCU_DSV4_PCP_EXPERIMENTAL", raising=False)

    assert _dsv4_pcp_experimental_enabled(config) is expected


@pytest.mark.parametrize(
    ("setting", "on_main_stream"),
    [(None, True), ("0", False), ("1", True)],
)
def test_output_copy_can_reuse_main_stream_without_initializing_device(
    monkeypatch, setting, on_main_stream
):
    main_stream, copy_stream = object(), object()
    config, device = object(), object()

    def fake_init(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.__dict__["main_stream"] = main_stream
        self.output_copy_stream = copy_stream

    monkeypatch.setattr(GPUModelRunner, "__init__", fake_init)
    if setting is None:
        monkeypatch.delenv("VLLM_HCU_MRV2_OUTPUT_COPY_ON_MAIN_STREAM", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_MRV2_OUTPUT_COPY_ON_MAIN_STREAM", setting)
    runner = HcuGPUModelRunnerV2(config, device)

    assert runner.output_copy_stream is (
        main_stream if on_main_stream else copy_stream
    )
    assert runner.vllm_config is config
    assert runner.device is device
    assert runner.pcp_manager is None
