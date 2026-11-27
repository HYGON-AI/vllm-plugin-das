"""CPU-only EP16 regression for HY4 MegaMoE collective scheduling."""
from datetime import timedelta
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_megamoe():
    return runpy.run_path(str(Path(__file__).resolve().parents[2]
                              / "vllm_hcu/models/hy_v4/megamoe.py"))


@pytest.mark.parametrize("rows", [0, 2])
def test_w4a8_launch_uses_predivided_scales_and_participates_when_empty(rows):
    run = _load_megamoe()["run_hy4_mega_experts"]
    provider = Mock()
    buf = SimpleNamespace(mode="w4a8", group=Mock(), handle=Mock(), buffer=object(),
                          local_buffer=object(), num_experts=272, num_topk=8,
                          num_max_tokens_per_rank=2560, _hy4_w4a8_launched=True)
    hidden = torch.empty((rows, 6144), dtype=torch.bfloat16)
    ids = torch.empty((rows, 8), dtype=torch.long)
    weights = torch.empty((rows, 8))
    gate_up, down, gate_up_scale, down_scale = [object() for _ in range(4)]
    with patch.dict(run.__globals__, _stage_w4a8=Mock()) as symbols:
        output = run(provider, hidden, ids, weights,
                     ((gate_up, gate_up_scale), (down, down_scale)), buf, 10.)
        assert symbols["_stage_w4a8"].call_count == int(rows > 0)
    assert output.shape == hidden.shape
    provider._C.w4a8_mega_moe.assert_called_once_with(
        output, gate_up, down, gate_up_scale, down_scale, buf.buffer, buf.local_buffer,
        buf.handle.buffer_ptrs, buf.handle.signal_ptrs, buf.group.rank(), 2560, 272, 8,
        10., True, l1_weights_sf_div16=gate_up_scale, l2_weights_sf_div16=down_scale,
    )
    provider.fp8_mega_moe.assert_not_called()


def _worker(rank, rendezvous):
    impl = _load_megamoe()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=16, timeout=timedelta(seconds=60))
    try:
        calls = []

        def run(provider, hidden, ids, weights, packed, buffer, clamp):
            assert len(hidden) == len(ids) == len(weights) <= buffer.num_max_tokens_per_rank
            calls.append(len(hidden))
            return hidden.clone()

        group = SimpleNamespace(cpu_group=dist.group.WORLD)
        kernels = [impl["Hy4MegaKernel"](None, None, SimpleNamespace(num_max_tokens_per_rank=2),
                                         group, 10) for _ in range(77)]
        with patch.dict(kernels[0].apply.__globals__, run_hy4_mega_experts=run), \
                patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
            for step, (rows, rounds) in enumerate((
                (0, 1), (2, 1), (5 if rank == 15 else 0, 3), (rank % 4, 2), (1, 1)
            ), start=1):
                hidden = torch.arange(rows * 8).reshape(rows, 8)
                ids = torch.zeros_like(hidden)
                calls.clear()
                with impl["mega_forward_batch"](rows):
                    for kernel in kernels:
                        out = kernel.apply(hidden, None, None, ids.float(), ids, None,
                                           272, None, False)
                        assert torch.equal(out, hidden)
                assert reduce.call_count == step
                assert calls == [min(2, max(rows - 2*i, 0)) for i in range(rounds)] * 77
    finally:
        dist.destroy_process_group()


def test_static_routing_preserves_logical_capture_and_chunk_offsets():
    impl = _load_megamoe()
    group = SimpleNamespace(cpu_group=None)
    kernel = impl["Hy4MegaKernel"](None, None, SimpleNamespace(num_max_tokens_per_rank=2), group, 10)
    hidden = torch.zeros(5, 8)
    ids = torch.arange(40).view(5, 8)
    weights = torch.ones(5, 8)
    router = Mock()
    router._compute_routing.return_value = weights, ids
    calls = []

    def launch(provider, values, logical, scores, packed, buffer, clamp, *, eplb, offset):
        assert eplb is router.eplb_state
        assert torch.equal(logical, ids[offset:offset + len(values)])
        calls.append(offset)
        return values.clone()

    with patch.dict(kernel.apply.__globals__, run_hy4_mega_experts=launch,
                    _chunk_rounds=lambda *args: 3):
        output = kernel.route_static(router, hidden, hidden, None)
    router._validate_eplb_state.assert_called_once_with()
    router.capture_fn.assert_called_once_with(ids)
    assert torch.equal(output, hidden)
    assert calls == [0, 2, 4]


def test_ep16_rounds_once_per_backbone_forward(tmp_path):
    mp.start_processes(_worker, args=((tmp_path / "gloo").as_uri(),),
                       nprocs=16, join=True, start_method="fork")


@pytest.mark.parametrize("pcp,dp", [(16, 1), (1, 16)])
@pytest.mark.parametrize("feature", ["DBO", "LoRA"])
def test_megamoe_rejects_unsupported_features_before_accessing_ep_group(monkeypatch, pcp, dp, feature):
    install_hy4_mega_kernel = _load_megamoe()["install_hy4_mega_kernel"]

    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["HYV4VLForConditionalGeneration"]),
        lora_config=SimpleNamespace() if feature == "LoRA" else None,
        parallel_config=SimpleNamespace(enable_dbo=feature == "DBO", prefill_context_parallel_size=pcp,
                                        data_parallel_size=dp),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    with patch("vllm.distributed.get_ep_group") as get_group:
        with pytest.raises(ValueError, match=f"MegaMoE does not support {feature}"):
            install_hy4_mega_kernel(None, None)
        get_group.assert_not_called()


@pytest.mark.parametrize("pcp,dp", [(16, 1), (1, 16)])
@pytest.mark.parametrize("placement,dtype,error", [
    ("round_robin", torch.bfloat16, "requires linear expert placement"),
    ("linear", torch.float16, "requires BF16 activations.*torch.float16"),
])
def test_megamoe_rejects_unsupported_expert_layout(monkeypatch, pcp, dp, placement, dtype, error):
    install = _load_megamoe()["install_hy4_mega_kernel"]
    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["HYV4VLForConditionalGeneration"], dtype=dtype),
        lora_config=None,
        parallel_config=SimpleNamespace(
            enable_dbo=False, prefill_context_parallel_size=pcp, data_parallel_size=dp,
            expert_placement_strategy=placement,
        ),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    with patch("vllm.distributed.get_ep_group") as get_group:
        with pytest.raises(ValueError, match=error):
            install(None, None)
        get_group.assert_not_called()
