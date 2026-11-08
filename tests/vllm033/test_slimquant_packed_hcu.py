# SPDX-License-Identifier: Apache-2.0
"""Public Core loader/native GPU Graph with independent signed INT4 CPU math."""

import inspect
import json
import os
from pathlib import Path

import pytest
from tests.vllm033.test_slimquant import checkpoint_config

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        os.environ.get("HCU_PACKED_LIVE") != "1",
        reason="explicit packed live validation",
    ),
]


def _unpack_reference(packed, *, swapped=False, unsigned=False):
    import torch

    codes = packed.to(torch.int16).bitwise_and(255)
    high, low = codes.floor_divide(16), codes.remainder(16)
    if not unsigned:
        high = torch.where(high >= 8, high - 16, high)
        low = torch.where(low >= 8, low - 16, low)
    pair = (low, high) if swapped else (high, low)
    return torch.stack(pair, dim=-1).flatten(-2).to(torch.int32)


def _quantized_reference(x, ids, routes, weights, dtype, **decode):
    """CPU integer GEMMs and explicit producer/storage rounding, no vendor API."""
    import torch
    import torch.nn.functional as F

    def quantize(value):
        value = value.float()
        scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        quantized = torch.round(value * scale.reciprocal()).clamp(-127, 127)
        return quantized.to(torch.int32), scale

    scale13, scale2, raw13, raw2 = [value for _, value in weights]
    xq, xs = quantize(x.cpu())
    ids, routes = ids.cpu(), routes.cpu()
    slots = torch.zeros((*ids.shape, x.shape[1]), dtype=dtype)
    for expert in range(raw13.shape[0]):
        rows, positions = torch.where(ids == expert)
        if not rows.numel():
            continue
        w13 = _unpack_reference(raw13[expert], **decode)
        w2 = _unpack_reference(raw2[expert], **decode)
        gate_up = (
            (xq[rows] @ w13.t()).float() * xs[rows] * (scale13[expert].flatten() * 16)
        ).to(dtype)
        gate, up = gate_up.float().chunk(2, dim=-1)
        activation = (F.silu(gate) * up).to(dtype)
        aq, activation_scale = quantize(activation)
        down = (
            (aq @ w2.t()).float() * activation_scale * (scale2[expert].flatten() * 16)
        )
        slots[rows, positions] = (down * routes[rows, positions, None]).to(dtype)
    result = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
    for slot in range(ids.shape[1]):
        result += slots[:, slot].float()
    return result.to(dtype)


@pytest.fixture
def actual_parallel(tmp_path):
    import torch
    from vllm.plugins import load_general_plugins
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        cleanup_dist_env_and_memory,
    )
    from vllm.distributed.parallel_state import set_custom_all_reduce
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    torch.cuda.set_device(0)
    load_general_plugins()
    with set_current_vllm_config(
        VllmConfig(
            parallel_config=ParallelConfig(disable_custom_all_reduce=True),
            additional_config={"hcu_slimquant": {"moe_provider": "boltops"}},
        )
    ):
        set_custom_all_reduce(False)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=(tmp_path / "dist-init").as_uri(),
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device("cuda:0"))
        try:
            yield
        finally:
            reset_workspace_manager()
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("padding", [False, True])
def test_actual_packed_constructor_loader_two_generations_and_graph(
    actual_parallel, dtype_name, padding, tmp_path
):
    import torch
    import vllm
    import hcu_vllm_das.slimquant as implementation
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    out = Path(os.environ.get("HCU_PACKED_RESULT_DIRECTORY", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, dtype_name)
    record_name = dtype_name + ("-padding" if padding else "")
    with torch.device("cuda:0"):
        runner = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=128,
            params_dtype=dtype,
            prefix="model.layers.0.mlp.experts",
            quant_config=implementation.SlimQuantConfig.from_config(
                checkpoint_config()
            ),
        )
    layer = runner.routed_experts
    method = layer.quant_method
    with torch.device("cuda:0"):
        record_metadata_for_reloading(layer)
    record = dict(
        dtype=dtype_name,
        padding=padding,
        method_type=type(method).__name__,
        module_origin=implementation.__file__,
        core_version=str(vllm.__version__),
        core_reload_origin=inspect.unwrap(initialize_layerwise_reload).__code__.co_filename,
        stage="SELECTED",
        full_model_accepted=False,
    )
    (out / f"{record_name}.json").write_text(json.dumps(record, indent=2) + "\n")
    assert type(method).__name__ == "SlimQuantPackedExperts", (
        "Native raw packed selector missing"
    )
    assert not method.supports_internal_mk and not method.is_monolithic
    assert tuple(layer.w13_weight.shape) == (4, 256, 128) and tuple(
        layer.w2_weight.shape
    ) == (4, 256, 64)
    pattern = torch.linspace(-0.125, 0.25, 256).repeat(3, 1).to(dtype)
    x = pattern.cuda()
    generator = torch.Generator().manual_seed(3381108)
    rejected = dict(swapped=False, unsigned=False)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32, device="cuda")
    routes = torch.tensor([[0.25, 0.75]] * 3, device="cuda")
    if padding:
        ids[0, 1] = -1
        ids[1].fill_(-1)
    pointers = [
        getattr(layer, n).data_ptr()
        for n in [
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_scale_for_kernel",
            "w2_scale_for_kernel",
        ]
    ]
    graph = None
    output = None
    cases = []
    for generation in range(2):
        scale = (torch.arange(1, 5, dtype=torch.float32)[:, None, None] / 4096) * (
            1 + generation * 0.5
        )
        weights = [
            ("gate_up_proj.weight_scale", scale.expand(4, 256, 1).contiguous()),
            ("down_proj.weight_scale", scale.expand(4, 256, 1).contiguous()),
            (
                "gate_up_proj",
                torch.randint(
                    -128, 128, (4, 256, 128), dtype=torch.int8, generator=generator
                ),
            ),
            (
                "down_proj",
                torch.randint(
                    -128, 128, (4, 256, 64), dtype=torch.int8, generator=generator
                ),
            ),
        ]
        if generation:
            initialize_layerwise_reload(layer)
        loaded = set(layer.load_weights(weights))
        assert loaded == {
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        }
        if generation:
            finalize_layerwise_reload(layer, model_config=None)
        else:
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)
        torch.testing.assert_close(
            layer.w13_weight_scale.cpu(), weights[0][1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.w13_scale_for_kernel.cpu(), weights[0][1] * 16, rtol=0, atol=0
        )
        assert pointers == [
            getattr(layer, n).data_ptr()
            for n in [
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_scale_for_kernel",
                "w2_scale_for_kernel",
            ]
        ]

        def apply():
            return method.apply(layer, x, routes, ids, None, None)

        if graph is None:
            for _ in range(3):
                apply()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = apply()
            address = output.data_ptr()
        for state in range(2):
            x.copy_(pattern.roll(state, 1) / (state + 1))
            if state:
                ids.copy_(ids.roll(1, 0))
                routes.copy_(routes.flip(1))
            if padding:
                ids.copy_(torch.tensor(
                    [[0, -1], [-1, -1], [2, 3]] if not state else
                    [[-1, -1], [1, 2], [-1, 3]], dtype=torch.int32, device="cuda"
                ))
                x.masked_fill_(~(ids >= 0).any(-1, keepdim=True), float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            reference = _quantized_reference(x, ids, routes, weights, dtype).double()
            for variant in rejected:
                wrong = _quantized_reference(
                    x, ids, routes, weights, dtype, **{variant: True}
                ).double()
                rejected[variant] |= not torch.allclose(
                    reference, wrong, rtol=0.015, atol=0.001
                )
            actual = output.cpu().double()
            difference = (actual - reference).abs()
            torch.testing.assert_close(actual, reference, rtol=0.015, atol=0.001)
            assert torch.isfinite(actual).all() and output.data_ptr() == address
            cases.append(
                dict(
                    generation=generation,
                    state=state,
                    max_abs=difference.max().item(),
                    relative_l2=(difference.norm() / reference.norm()).item(),
                )
            )
    assert all(rejected.values()), (
        "fixture must distinguish nibble order and signed interpretation"
    )
    record.update(
        stage="PASS_NATIVE_GPU_CONSTRUCTOR_OFFICIAL_RAW_RELOAD_TWO_GENERATIONS_GRAPH",
        official_layerwise_reload=True,
        cases=cases,
        wrong_decode_variants_rejected=rejected,
        native_provider=method._provider.__module__,
        raw_dtype=str(layer.w13_weight.dtype),
        canonical_scale_dtype=str(layer.w13_weight_scale.dtype),
        six_pointers_stable=True,
        graph_output_pointer_stable=True,
        scope="Actual TP1,4experts,signed/nonuniform quantized CPU references; not actualTP2/4 or fullcheckpoint/provider/performance acceptance",
    )
    (out / f"{record_name}.json").write_text(json.dumps(record, indent=2) + "\n")
