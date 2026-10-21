"""CPU policy and event-order contracts; no CUDA kernels are launched."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import shared_experts as runtime
from vllm_hcu.platforms import envs as henvs


@pytest.fixture
def shared_setup(monkeypatch):
    calls = []
    class Stream:
        def __init__(self, name):
            self.name = name
        def wait_stream(self, other):
            calls.append(("wait", self.name, other.name))
    main, auxiliary = Stream("main"), Stream("aux")
    monkeypatch.setattr(runtime, "current_stream", lambda: main)
    monkeypatch.setattr(runtime, "aux_stream", lambda: auxiliary)
    monkeypatch.setattr(runtime, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setattr(runtime.envs, "VLLM_DISABLE_SHARED_EXPERTS_STREAM", False)
    monkeypatch.setattr(runtime.envs, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 8)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE", False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream:
                        calls.append(("record", stream.name)))
    @contextmanager
    def stream_scope(stream):
        calls.append(("enter", stream.name))
        yield
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    class Layer(torch.nn.Module):
        def forward(self, x):
            calls.append(("compute",))
            return x * 3 + 1
    def build(backend="deepep_auto", safe=True, dbo=False, eplb=True, fi=False, mk_overlap=True):
        config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
            enable_eplb=eplb, all2all_backend=backend,
            use_fi_nvl_two_sided_kernels=fi,
        ))
        return runtime.SharedExperts(Layer(), config, dbo, lambda: mk_overlap, lambda: safe)
    return build, calls


@pytest.mark.parametrize("backend", ["deepep_auto", "deepep_high_throughput"])
@pytest.mark.parametrize("mk_overlap", [True, False])
def test_early_launch_joins_only_when_consumed(shared_setup, backend, mk_overlap):
    build, calls = shared_setup
    shared = build(backend=backend, mk_overlap=mk_overlap)
    x = torch.arange(8).float().view(2, 4)
    for iteration in range(2):
        calls.clear()
        shared.maybe_sync_shared_experts_stream(x)
        assert shared._determine_shared_experts_order(x) is runtime.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        assert calls[:3] == [("record", "aux"), ("wait", "aux", "main"), ("enter", "aux")]
        assert calls.count(("compute",)) == 1
        assert ("wait", "main", "aux") not in calls
        # Runner/MK probes other orders; none may recompute early output.
        for order in runtime.SharedExpertsOrder:
            if order is not runtime.SharedExpertsOrder.NONE:
                shared(x, order)
        assert calls.count(("compute",)) == 1
        torch.testing.assert_close(shared.output, x * 3 + 1)
        assert calls.count(("wait", "main", "aux")) == 1
        assert shared._output[0] is None
        assert shared._output_pending_on_stream[0] is False


@pytest.mark.parametrize("settings", [
    {"safe": False}, {"dbo": True}, {"backend": "deepep_low_latency"},
    {"backend": "allgather_reducescatter", "fi": True},
    {"backend": "pplx"}, {"backend": "deepep_auto", "fi": True},
])
def test_unsupported_eplb_paths_stay_serial(shared_setup, settings):
    build, calls = shared_setup
    shared = build(**settings)
    x = torch.ones(2, 4)
    assert not shared._can_overlap_deepep_eplb()
    assert shared._determine_shared_experts_order(x) is runtime.SharedExpertsOrder.NO_OVERLAP
    shared.maybe_sync_shared_experts_stream(x)
    assert not calls
    shared(x, runtime.SharedExpertsOrder.NO_OVERLAP)
    torch.testing.assert_close(shared.output, x * 3 + 1)
    assert not any(call[0] == "wait" for call in calls)


@pytest.mark.parametrize("switch", ["child", "master", "stream", "threshold"])
def test_overlap_rollback_and_resource_guards(shared_setup, monkeypatch, switch):
    build, calls = shared_setup
    shared = build()
    if switch == "child":
        monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP", False)
    elif switch == "master":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    elif switch == "stream":
        shared._stream = None
    else:
        monkeypatch.setattr(runtime.envs, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 1)
    x = torch.ones(2, 4)
    expected = (runtime.SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
                if switch == "threshold" else runtime.SharedExpertsOrder.NO_OVERLAP)
    assert shared._determine_shared_experts_order(x) is expected
    shared.maybe_sync_shared_experts_stream(x)
    assert not calls
    shared(x, expected)
    torch.testing.assert_close(shared.output, x * 3 + 1)


def test_eplb_overlap_preserves_raw_input_alias(shared_setup):
    build, _ = shared_setup
    shared = build()
    x = torch.ones(2, 4)
    assert shared.requires_input_preservation(x)
    assert not shared.allows_inplace_routed_output(x, x.view_as(x))
    assert shared.allows_inplace_routed_output(x.clone(), x)
