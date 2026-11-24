"""Step5-only scope and the routed TP transport's reconstruction contract."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime as runtime
from vllm_hcu.runtime_compat.step5 import is_step5_tp4_deepep_ll
from vllm_hcu.runtime_compat.step5 import configure_step5_deepep_hca


def config(model_type="step5", tp=4, dp=4, backend="deepep_low_latency", sp=False, dbo=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp,
            enable_expert_parallel=True, enable_dbo=dbo, pipeline_parallel_size=1,
            all2all_backend=backend),
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=sp)))


@pytest.mark.parametrize("kwargs", [dict(model_type="qwen2_moe"), dict(tp=8, dp=2),
    dict(tp=1, dp=16), dict(backend="allgather_reducescatter"), dict(sp=True), dict(dbo=True)])
def test_other_routes_retain_their_transport(kwargs):
    assert not is_step5_tp4_deepep_ll(config(**kwargs))


def test_only_the_requested_step5_route_is_selected():
    assert is_step5_tp4_deepep_ll(config())
    assert not is_step5_tp4_deepep_ll(SimpleNamespace(model_type="step5"))


@pytest.mark.parametrize("rank", range(8))
def test_nic_mapping_uses_logical_rank_with_reordered_physical_devices(monkeypatch, rank):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3,4,6,7,5")
    monkeypatch.setenv("DEEP_EP_DEVICE_TO_HCA_MAPPING", ",".join(f"{i}:mlx5_{i}:1" for i in range(8)))
    monkeypatch.setenv("ROCSHMEM_USE_IB_HCA", "mlx5_0")
    configure_step5_deepep_hca(config(), rank)
    import os
    assert os.environ["ROCSHMEM_USE_IB_HCA"] == f"mlx5_{rank}"


@pytest.mark.parametrize("kwargs", [dict(model_type="qwen2_moe"), dict(tp=8, dp=2),
                                    dict(backend="allgather_reducescatter")])
def test_other_routes_ignore_step5_nic_mapping(monkeypatch, kwargs):
    monkeypatch.setenv("DEEP_EP_DEVICE_TO_HCA_MAPPING", "invalid")
    monkeypatch.setenv("ROCSHMEM_USE_IB_HCA", "original")
    configure_step5_deepep_hca(config(**kwargs), 7)
    import os
    assert os.environ["ROCSHMEM_USE_IB_HCA"] == "original"


@pytest.mark.parametrize("n", [4, 16, 64])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("event_present", [False, True])
def test_dispatch_shards_and_combine_restores_full_rows(monkeypatch, n, rank, event_present):
    import vllm.distributed as distributed
    import vllm.distributed.device_communicators.pynccl as pynccl

    hidden = 8
    source = torch.arange(n * hidden).reshape(n, hidden).to(torch.bfloat16)
    ids = torch.arange(n * 8).reshape(n, 8) % 352
    weights = torch.full((n, 8), .125)
    expected = (source.float() + ids.sum(1, keepdim=True).float()).to(torch.bfloat16)
    order, calls = [], []
    group = SimpleNamespace(world_size=4, rank_in_group=rank, cpu_group=object())
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "current-stream")
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: empty(*a, **{**k, "device": "cpu"}))

    class Comm:
        available, disabled = True, False

        def __init__(self, *args, **kwargs):
            pass

        def all_gather(self, output, local, stream):
            assert stream == "current-stream"
            assert order == (["receive", "event_wait"] if event_present else ["receive"])
            torch.testing.assert_close(local, expected.chunk(4)[rank], rtol=0, atol=0)
            order.append("tp_gather")
            output.copy_(expected)

    monkeypatch.setattr(pynccl, "PyNcclCommunicator", Comm)

    class Event:
        event = object() if event_present else None

        def current_stream_wait(self):
            assert self.event is not None
            order.append("event_wait")

    native_handle = object()

    class Buffer:
        group_size = 16

        def low_latency_dispatch(self, x, i, w, *args, **kwargs):
            assert kwargs["return_recv_hook"] is False
            torch.testing.assert_close(x, source.chunk(4)[rank], rtol=0, atol=0)
            assert torch.equal(i, ids.chunk(4)[rank]) and torch.equal(w, weights.chunk(4)[rank])
            return x, "counts", native_handle, None, None

        def low_latency_combine(self, x, i, w, handle, **kwargs):
            assert kwargs["return_recv_hook"] is False
            assert handle is native_handle
            assert torch.equal(i, ids.chunk(4)[rank])
            assert torch.equal(w, weights.chunk(4)[rank])
            calls.append(x)

            def receive():
                order.append("receive")
                kwargs["out"].copy_(expected.chunk(4)[rank])

            receive()
            return kwargs["out"], Event(), None

    proxy = runtime._Step5TPTokenBuffer(Buffer(), 64, hidden)
    _, counts, handle, _, _ = proxy.low_latency_dispatch(source, ids, weights, 64, 352)
    assert counts == "counts"
    output = torch.empty_like(expected)
    experts = torch.empty((22, 3, hidden), dtype=torch.bfloat16)
    actual, _, hook = proxy.low_latency_combine(
        experts, ids, weights, handle, out=output, async_finish=False, return_recv_hook=True)
    assert len(calls) == 1 and calls[0] is experts
    expected_order = ["receive", "event_wait", "tp_gather"] if event_present else ["receive", "tp_gather"]
    assert order == expected_order
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    hook()
    assert order == expected_order


@pytest.mark.parametrize("n", [1, 2, 7])
def test_unpartitioned_tails_preserve_tokens_and_native_handle(n):
    handle, event, hook = object(), SimpleNamespace(event=None), None
    source = torch.ones((n, 8), dtype=torch.bfloat16)
    ids = torch.zeros((n, 8), dtype=torch.int64)
    weights = torch.ones((n, 8))
    result = (source, event, hook)

    class Buffer:
        def low_latency_dispatch(self, x, i, w, *args, **kwargs):
            assert x is source and i is ids and w is weights
            return x, "counts", handle, event, hook

        def low_latency_combine(self, x, i, w, native, **kwargs):
            assert native is handle and i is ids and w is weights
            return result

        def get_next_low_latency_combine_buffer(self, native):
            assert native is handle
            return "next"

    proxy = object.__new__(runtime._Step5TPTokenBuffer)
    proxy._buffer, proxy._size, proxy._rank = Buffer(), 4, 1
    received = proxy.low_latency_dispatch(source, ids, weights, 64, 352)
    assert received[2] == (handle, None)
    actual = proxy.low_latency_combine(source, ids, weights, received[2])
    assert actual[0] is source and actual[1] is event and actual[2] is None
    assert proxy.get_next_low_latency_combine_buffer(received[2]) == "next"


def test_async_combine_is_rejected_before_native_side_effects():
    proxy = object.__new__(runtime._Step5TPTokenBuffer)
    proxy._size, proxy._rank = 4, 0
    proxy._buffer = SimpleNamespace(low_latency_combine=lambda *a, **k: pytest.fail("native call"))
    with pytest.raises(ValueError, match="synchronous combine"):
        proxy.low_latency_combine(torch.empty((22, 3, 8)), None, None,
                                  (object(), 16), async_finish=True)


def test_async_dispatch_is_rejected_before_native_side_effects():
    proxy = object.__new__(runtime._Step5TPTokenBuffer)
    proxy._buffer = SimpleNamespace(low_latency_dispatch=lambda *a, **k: pytest.fail("native call"))
    with pytest.raises(ValueError, match="synchronous dispatch"):
        proxy.low_latency_dispatch(None, None, None, 512, 352, async_finish=True)


@pytest.mark.parametrize("cold_capture", [False, True])
def test_pure_ll_initializes_once_before_graph_capture(monkeypatch, cold_capture):
    import vllm.distributed as distributed
    import torch.distributed as dist

    capturing = [cold_capture]
    calls = []
    monkeypatch.setattr(distributed, "get_ep_group", lambda: SimpleNamespace(cpu_group="ep-cpu"))
    monkeypatch.setattr(dist, "barrier", lambda **k: calls.append("barrier"))
    monkeypatch.setattr(runtime, "_claim_ll_buffer_cleanup",
                        lambda *a: pytest.fail("Pure LL must not use per-forward HT/LL cleanup"))
    fake_cuda = SimpleNamespace(is_current_stream_capturing=lambda: capturing[0],
                                synchronize=lambda: calls.append("synchronize"))
    module = SimpleNamespace(torch=SimpleNamespace(cuda=fake_cuda), dbo_current_ubatch_id=lambda: 0)
    buffer = SimpleNamespace(clean_low_latency_buffer=lambda *a: calls.append("clean"),
                             low_latency_dispatch=lambda *a, **k: (None, None, "handle", None, "hook"))
    prepare = SimpleNamespace(_hcu_low_latency_dispatch_abi=True, use_int8_dispatch=False,
        _hcu_low_latency_cleanup_topk=False,
        use_fp8_dispatch=False, use_ue8m0_dispatch=False, SUPPORTED_HIDDEN_SIZES={4096},
        _vllm_hcu_clean_low_latency_buffer=True, _hcu_step5_ll_transport=True,
        buffer=buffer, max_tokens_per_rank=512, _map_global_to_physical_ids=lambda x:x,
        handles={})
    states = torch.empty((16,4096), dtype=torch.bfloat16)
    ids = torch.zeros((16,8), dtype=torch.int64)
    weights = torch.ones((16,8))
    quant = SimpleNamespace(quant_dtype="fp8", a1_scale=None, a2_scale=None)

    def invoke():
        return runtime.ll_prepare_async(module, None, prepare, states, weights, ids,
                                        352, None, False, quant)

    if cold_capture:
        with pytest.raises(RuntimeError, match="before graph capture"):
            invoke()
        assert calls == [] and not prepare.handles
    else:
        assert invoke()[0] == "hook"
        capturing[0] = True
        assert invoke()[0] == "hook"
        assert calls == ["clean", "synchronize", "barrier"]
        assert buffer._hcu_step5_ll_initialized is True


@pytest.mark.parametrize("step5_owner", [False, True])
def test_execution_keeps_construction_owner_after_config_context_exits(monkeypatch, step5_owner):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.model_executor.layers.fused_moe.experts import dpsk_v4_deep_gemm_moe as experts

    instance = object.__new__(experts.DeepEPDeepGemmMaskedExperts)
    instance._step5_ll = step5_owner
    monkeypatch.setattr(experts, "is_step5_tp4_deepep_ll", lambda: False)
    monkeypatch.setattr(experts.henvs, "optional_custom_op_enabled", lambda value: bool(value))
    args = dict(output=None, hidden_states=None, w1=None, w2=None, topk_weights=None,
                topk_ids=None, activation=MoEActivation.SWIGLUSTEP, global_num_experts=352,
                expert_map=None, a1q_scale=None, a2_scale=None, workspace13=None,
                workspace2=None, expert_tokens_meta=None, apply_router_weight_on_input=False)
    if step5_owner:
        with pytest.raises(RuntimeError, match="expert token metadata"):
            instance.apply(**args)
    else:
        with pytest.raises(NotImplementedError, match="SWIGLUSTEP"):
            instance.apply(**args)


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(),
                    reason="Requires ROCm FP8 and GPU graph capture")
def test_masked_activation_matches_native_with_new_graph_routing():
    from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import (
        ep_masked_swiglustep_quant_deepgemm,
        ep_swiglustep_quant,
    )

    source = torch.randn((3, 8, 3072), dtype=torch.bfloat16, device="cuda") * 12
    counts = torch.tensor([0, 2, 8], dtype=torch.int32, device="cuda")
    q = torch.empty((3, 8, 1536), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty((3, 8, 1), device="cuda")

    def launch():
        ep_masked_swiglustep_quant_deepgemm(source, counts, q=q, scale=scale)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for lengths in ([0, 2, 8], [8, 3, 0]):
        source.normal_().mul_(12)
        counts.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        stream.wait_stream(torch.cuda.current_stream())
        graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        valid = torch.cat([source[e, :n] for e, n in enumerate(lengths) if n])
        rows = valid.shape[0]
        expected_q, expected_scale = ep_swiglustep_quant(
            valid, torch.ones(rows, device="cuda"),
            torch.tensor([rows], dtype=torch.int32, device="cuda"), rows)
        actual_q = torch.cat([q[e, :n] for e, n in enumerate(lengths) if n])
        actual_scale = torch.cat([scale[e, :n] for e, n in enumerate(lengths) if n])
        assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
