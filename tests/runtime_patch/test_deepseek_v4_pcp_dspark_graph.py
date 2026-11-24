# SPDX-License-Identifier: Apache-2.0
"""PCP decode graph eligibility and capture ownership."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from tests.runtime_patch.test_glm52_pcp_config import _make_pcp_config
from tests.runtime_patch.test_glm52_pcp_runner import _config, pcp_runner_module
from vllm_hcu.patch.platform.core_fix.patch_vllm_config import _validate_hcu_pcp_scope


@pytest.mark.parametrize("requested", [
    CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
])
def test_dspark_pcp_normalizes_to_decode_graphs(requested):
    config = _make_pcp_config(
        architecture="DeepseekV4ForCausalLM", tp=1, pcp=8,
        speculative=True, speculative_method="dspark",
        num_speculative_tokens=7, enforce_eager=False,
    )
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=requested, mode=CompilationMode.VLLM_COMPILE,
    )
    assert _validate_hcu_pcp_scope(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert config.compilation_config.mode == CompilationMode.NONE


def test_pcp_preserves_explicit_graph_disable():
    config = _make_pcp_config(
        architecture="DeepseekV4ForCausalLM", speculative=True,
        speculative_method="dspark", enforce_eager=False,
    )
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE, mode=CompilationMode.NONE,
    )
    assert _validate_hcu_pcp_scope(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("phase", [[], [False], [True], [False, True]])
def test_prefill_never_replays_decode_graph(phase):
    from vllm_hcu.v1.deepseek_v4_pcp_graph import install_decode_dispatch

    calls = []
    expected = object()
    manager = SimpleNamespace(dispatch=lambda *args: (calls.append(args), expected)[1])
    runner = SimpleNamespace(cudagraph_manager=manager,
                             _hcu_pcp_graph_has_prefill=any(phase))
    install_decode_dispatch(runner)
    install_decode_dispatch(runner)
    result = manager.dispatch(1, 8, 8)
    if any(phase):
        assert result.cg_mode == CUDAGraphMode.NONE
        assert result.num_tokens == 8
        assert not calls
    else:
        assert result is expected
        assert len(calls) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_capture_uses_global_buffers_and_restores_manager(
    pcp_runner_module, monkeypatch, fail,
):
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch

    module, _ = pcp_runner_module
    config = _config(8)
    config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    config.speculative_config = SimpleNamespace(method="dspark")
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    manager = runner.pcp_manager = object()
    buffers = runner.input_buffers = object()
    def capture(self, *, profile_only):
        assert self.pcp_manager is None
        assert self.input_buffers is buffers
        assert in_replicated_mtp_batch()
        assert profile_only
        if fail:
            raise RuntimeError("capture failed")
        return 123
    monkeypatch.setattr(module.GPUModelRunner, "capture_model", capture, raising=False)
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            runner.capture_model(profile_only=True)
    else:
        assert runner.capture_model(profile_only=True) == 123
    assert runner.pcp_manager is manager
    assert not in_replicated_mtp_batch()


def test_dispatch_phase_is_global_and_refreshed(pcp_runner_module, monkeypatch):
    module, _ = pcp_runner_module
    config = _config(8)
    config.model_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    config.speculative_config = SimpleNamespace(method="dspark")
    runner = module.HcuGPUModelRunnerV2(config, "cpu")
    for phase in ([False, True], [False, False]):
        result = (SimpleNamespace(is_prefilling_np=np.array(phase)), None)
        monkeypatch.setattr(module.GPUModelRunner, "gather_batch_req_state",
                            lambda *args: result, raising=False)
        assert runner.gather_batch_req_state(None) is result
        assert runner._hcu_pcp_graph_has_prefill == any(phase)


@pytest.mark.parametrize("num_reqs", [1, 3, 7, 15, 16])
def test_decode_metadata_padding_preserves_live_rows(num_reqs):
    from dataclasses import dataclass
    import torch
    from vllm_hcu.v1.deepseek_v4_pcp_graph import prepare_decode_graph_metadata

    @dataclass
    class Batch:
        num_reqs_after_padding: int
        num_tokens: int
        num_tokens_after_padding: int
        is_prefilling_np: np.ndarray

    batch = Batch(16, num_reqs * 8, 128, np.zeros(num_reqs, dtype=bool))
    canonical = torch.full((2, 256), 123, dtype=torch.int64)
    replica = canonical.clone()
    manager = SimpleNamespace(
        _global_slot_mappings=replica,
        _block_tables=SimpleNamespace(slot_mappings=canonical),
        _pad_slot_id=-1,
    )
    padded, slots = prepare_decode_graph_metadata(manager, batch)
    assert batch.is_prefilling_np.size == num_reqs
    assert padded.is_prefilling_np.size == 16
    assert not padded.is_prefilling_np.any()
    assert slots.shape == (2, 128)
    for buffer in (canonical, replica):
        assert (buffer[:, :batch.num_tokens] == 123).all()
        assert (buffer[:, batch.num_tokens:128] == -1).all()
        assert (buffer[:, 128:] == 123).all()


@pytest.mark.parametrize("draft_width", [5, 7])
def test_draft_graphs_cover_target_request_buckets_without_mutating_config(draft_width):
    from vllm_hcu.v1.deepseek_v4_pcp_graph import align_draft_graph_buckets

    class DraftManager:
        def __init__(self, config, device, mode, *, decode_query_len):
            self.vllm_config = config
            self.compilation_config = config.compilation_config
            self.cudagraph_mode = mode
            self.width = decode_query_len

    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[8, 16, 32, 64, 128],
        max_cudagraph_capture_size=128,
    )
    config = SimpleNamespace(compilation_config=compilation)
    old = DraftManager(config, "cpu", CUDAGraphMode.FULL_DECODE_ONLY,
                       decode_query_len=draft_width)
    speculator = SimpleNamespace(query_cudagraph_manager=old,
                                 num_query_per_req=draft_width)
    runner = SimpleNamespace(
        speculator=speculator, decode_query_len=8, max_num_reqs=64,
        cudagraph_manager=SimpleNamespace(compilation_config=compilation),
        device="cpu",
    )
    align_draft_graph_buckets(runner)
    new = speculator.query_cudagraph_manager
    assert new.compilation_config.cudagraph_capture_sizes == [
        n * draft_width for n in (1, 2, 4, 8, 16)
    ]
    assert new.compilation_config.max_cudagraph_capture_size == 16 * draft_width
    assert new.vllm_config is not config
    assert compilation.cudagraph_capture_sizes == [8, 16, 32, 64, 128]
    assert compilation.max_cudagraph_capture_size == 128


@pytest.mark.parametrize("deepep_auto", [False, True])
def test_pcp_piecewise_request_never_upgrades_to_full(deepep_auto):
    from vllm_hcu.patch.config import HcuFeatureConfig
    config = _make_pcp_config(
        architecture="DeepseekV4ForCausalLM", tp=1, pcp=8,
        speculative=True, speculative_method="dspark",
        num_speculative_tokens=5, enforce_eager=False,
        all2all_backend="deepep_low_latency",
    )
    config.additional_config = {"hcu": HcuFeatureConfig(deepep_auto=deepep_auto).to_dict()}
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.PIECEWISE, mode=CompilationMode.VLLM_COMPILE,
    )
    with pytest.raises(ValueError, match="PIECEWISE"):
        _validate_hcu_pcp_scope(config)
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode.PIECEWISE
