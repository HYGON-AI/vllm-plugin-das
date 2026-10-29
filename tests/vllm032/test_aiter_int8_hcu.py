from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.hcu

SUPPORTED_ARCHITECTURES = frozenset({"gfx936", "gfx938"})
EXPERTS = 256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
TOP_K = 8
INT8_LIMITS = {
    "min_reference_rms": 1e-3,
    "max_nmae": 3e-2,
    "max_nrmse": 3e-2,
    "max_abs_error": 1e-3,
}


def _live_requested() -> bool:
    return os.environ.get("HCU_VLLM032_AITER_INT8_LIVE") == "1"


def _assert_int8_numerics(actual, reference) -> dict[str, float]:
    import torch

    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.isfinite(reference).all()
    assert torch.count_nonzero(actual).item() > 0

    reference_float = reference.float()
    error = actual.float() - reference_float
    reference_rms = reference_float.square().mean().sqrt()
    reference_mean_abs = reference_float.abs().mean()
    nmae = error.abs().mean() / reference_mean_abs
    nrmse = error.square().mean().sqrt() / reference_rms
    max_abs_error = error.abs().max()
    metrics = {
        "reference_rms": reference_rms.item(),
        "nmae": nmae.item(),
        "nrmse": nrmse.item(),
        "max_abs_error": max_abs_error.item(),
    }
    assert metrics["reference_rms"] >= INT8_LIMITS["min_reference_rms"]
    assert metrics["nmae"] <= INT8_LIMITS["max_nmae"]
    assert metrics["nrmse"] <= INT8_LIMITS["max_nrmse"]
    assert metrics["max_abs_error"] <= INT8_LIMITS["max_abs_error"]
    return metrics


def _inputs(torch, m: int, device):
    generator = torch.Generator(device=device).manual_seed(20261029 + m)
    hidden_states = torch.randn(
        (m, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    topk_ids = (
        torch.arange(m * TOP_K, device=device).reshape(m, TOP_K) % EXPERTS
    ).to(torch.int32)
    topk_weights = torch.rand(
        (m, TOP_K),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_weights, topk_ids


@pytest.fixture(scope="module")
def qwen_int8_tensors():
    if not _live_requested():
        pytest.skip(
            "set HCU_VLLM032_AITER_INT8_LIVE=1 for live AITER INT8 MoE"
        )

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm/HCU device is unavailable")
    device = torch.device("cuda", 0)
    architecture = torch.cuda.get_device_properties(device).gcnArchName.split(
        ":", 1
    )[0]
    assert architecture in SUPPORTED_ARCHITECTURES

    generator = torch.Generator(device=device).manual_seed(20261029)
    w1 = torch.randint(
        -16,
        17,
        (EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    w2 = torch.randint(
        -16,
        17,
        (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    w1_scale = torch.rand(
        (EXPERTS, 2 * INTERMEDIATE_SIZE, 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.006 + 0.0015
    w2_scale = torch.rand(
        (EXPERTS, HIDDEN_SIZE, 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.006 + 0.0015
    return SimpleNamespace(
        architecture=architecture,
        device=device,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.skipif(
    not _live_requested(),
    reason="set HCU_VLLM032_AITER_INT8_LIVE=1 for live AITER INT8 MoE",
)
@pytest.mark.parametrize("m", (1, 16, 128))
def test_aiter_int8_matches_triton_and_replays_graph(
    qwen_int8_tensors,
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    import torch

    from aiter import moe as aiter_moe
    from vllm.model_executor.layers.fused_moe.config import (
        int8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl,
    )

    from hcu_vllm_das.moe import fused_moe_aiter as provider

    tensors = qwen_int8_tensors
    hidden_states, topk_weights, topk_ids = _inputs(
        torch, m, tensors.device
    )
    quant_config = int8_w8a8_moe_quant_config(
        w1_scale=tensors.w1_scale,
        w2_scale=tensors.w2_scale,
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    moe_config = SimpleNamespace(num_experts=EXPERTS)

    found, selected = aiter_moe.get_aiter_moe_config(
        m,
        EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        TOP_K,
        0,
        torch.bfloat16,
        aiter_moe.MoeQuantType.W8A8,
        activation="silu",
        use_shuffle=0,
    )
    assert found, f"no AITER W8A8 config for M={m}"
    assert not selected.need_shuffle
    assert not selected.need_shuffle_scale

    hits = 0
    original_api = provider._load_aiter_api()

    def counted_aiter_moe(**kwargs):
        nonlocal hits
        hits += 1
        return original_api.aiter_moe(**kwargs)

    monkeypatch.setattr(
        provider,
        "_load_aiter_api",
        lambda: SimpleNamespace(
            quant_type_w8a8=original_api.quant_type_w8a8,
            get_config=original_api.get_config,
            aiter_moe=counted_aiter_moe,
        ),
    )

    def optimized(value):
        return provider.apply_aiter_int8_moe(
            hidden_states=value,
            w1=tensors.w1,
            w2=tensors.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_config,
            quant_config=quant_config,
            activation="silu",
            expert_map=None,
            output_dtype=torch.bfloat16,
        )

    reference = fused_experts_impl(
        hidden_states=hidden_states,
        w1=tensors.w1,
        w2=tensors.w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        global_num_experts=EXPERTS,
        expert_map=None,
        w1_scale=tensors.w1_scale,
        w2_scale=tensors.w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    actual = optimized(hidden_states)
    torch.cuda.synchronize()
    metrics = _assert_int8_numerics(actual, reference)

    for _ in range(3):
        optimized(hidden_states)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = optimized(hidden_states)
    torch.cuda.synchronize()
    captured_output = graph_output.clone()

    changed_hidden = hidden_states + 0.03125
    eager_changed = optimized(changed_hidden)
    hidden_states.copy_(changed_hidden)
    graph.replay()
    torch.cuda.synchronize()
    replayed = graph_output.clone()

    assert not torch.equal(replayed, captured_output)
    replay_metrics = _assert_int8_numerics(replayed, eager_changed)
    assert hits >= 6
    print(
        {
            "arch": tensors.architecture,
            "m": m,
            "solution": str(selected.solution_type),
            "config": selected.config,
            "triton_metrics": metrics,
            "graph_metrics": replay_metrics,
            "aiter_kernel_calls": hits,
        }
    )
