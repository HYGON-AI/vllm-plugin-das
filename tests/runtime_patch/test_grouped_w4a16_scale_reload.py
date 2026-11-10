# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Captured grouped GPTQ scales must follow weight-update lifecycles."""

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import grouped_w4a16_moe_runtime


def _apply_grouped(experts):
    device = experts.w1_scale.device
    return grouped_w4a16_moe_runtime.apply_grouped_w4a16_moe(
        experts=experts,
        output=torch.empty(3, 256, dtype=torch.bfloat16, device=device),
        hidden_states=torch.ones(3, 256, dtype=torch.bfloat16, device=device),
        w1=torch.ones(2, 256, 128, dtype=torch.uint8, device=device),
        w2=torch.ones(2, 256, 64, dtype=torch.uint8, device=device),
        topk_weights=torch.ones(3, 1, device=device),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32, device=device),
        activation="silu", global_num_experts=2, expert_map=None,
        a1q_scale=None, a2_scale=None, expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )


def _captured_grouped_model(monkeypatch, device="cpu"):
    aiter = ModuleType("aiter.moe")
    aiter.MoeQuantType = SimpleNamespace(W4A16="w4a16")
    aiter.aiter_moe_shfl_scale = lambda s1, s2, config: (s1.clone(), s2.clone())
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter)
    config = SimpleNamespace(
        solution_type="moe_c", quant_type="w4a16", need_shuffle=False,
        need_shuffle_scale=True, block_size=128, config={},
    )
    monkeypatch.setattr(
        grouped_w4a16_moe_runtime, "select_aiter_moe_config",
        lambda problem, cache_owner: config,
    )
    captured = []

    def execute(config, **kwargs):
        captured.append((kwargs["w1_scale"], kwargs["w2_scale"]))
        return kwargs["hidden_states"]

    monkeypatch.setattr(grouped_w4a16_moe_runtime, "execute_aiter_moe", execute)
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_scales",
        torch.nn.Parameter(torch.ones(2, 256, 2, device=device), requires_grad=False),
    )
    layer.register_parameter(
        "w2_scales",
        torch.nn.Parameter(torch.ones(2, 256, 1, device=device), requires_grad=False),
    )
    quant = SimpleNamespace(use_int4_w4a16=True, w1_zp=None, w2_zp=None)
    experts = SimpleNamespace(
        quant_config=quant, block_shape=[0, 128],
        moe_config=SimpleNamespace(moe_backend="aiter"),
        w1_scale=layer.w13_scales, w2_scale=layer.w2_scales,
    )
    method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=experts))
    layer.quant_method = SimpleNamespace(old_quant_method=method)
    with torch.inference_mode():
        assert _apply_grouped(experts)
    assert len(captured) == 1
    return torch.nn.Sequential(layer), layer, method, captured[0], captured


@pytest.mark.parametrize("update", ["copy", "data"])
def test_finish_weight_update_refreshes_captured_grouped_scales_without_eager(
    monkeypatch, update,
):
    from vllm.v1.worker.gpu_worker import Worker
    from vllm_hcu.v1.worker import HcuGPUWorker

    model, layer, _, captured, _ = _captured_grouped_model(monkeypatch)
    addresses = tuple(scale.data_ptr() for scale in captured)

    def write_scales(self):
        for source in (layer.w13_scales, layer.w2_scales):
            with torch.no_grad():
                if update == "data":
                    source.data.copy_(torch.full_like(source, 3))
                else:
                    source.copy_(torch.full_like(source, 3))

    monkeypatch.setattr(Worker, "finish_weight_update", write_scales)
    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(model=model, get_draft_model=lambda: None)
    worker._weight_update_is_draft = False
    worker.finish_weight_update()

    assert tuple(scale.data_ptr() for scale in captured) == addresses
    for scale in captured:
        torch.testing.assert_close(scale, torch.full_like(scale, 3))


@pytest.mark.parametrize("is_draft", [False, True])
def test_weight_transfer_snapshots_owner_before_kernel_rebuild(
    monkeypatch, is_draft,
):
    from vllm.v1.worker.gpu_worker import Worker
    from vllm_hcu.v1.worker import HcuGPUWorker

    model, layer, method, captured, _ = _captured_grouped_model(monkeypatch)
    starts = iter((None, RuntimeError("already active")))

    def start(self, is_draft=False):
        error = next(starts)
        if error is not None:
            raise error

    monkeypatch.setattr(Worker, "_start_weight_update", start)
    monkeypatch.setattr(Worker, "finish_weight_update", lambda self: None)
    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Module() if is_draft else model,
        get_draft_model=lambda: model if is_draft else None,
    )
    worker._weight_update_is_draft = is_draft
    worker._start_weight_update(is_draft=is_draft)

    for source in (layer.w13_scales, layer.w2_scales):
        with torch.no_grad():
            source.copy_(torch.full_like(source, 3))
    old_experts = method.moe_kernel.fused_experts
    method.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(
        quant_config=SimpleNamespace(use_int4_w4a16=True),
        w1_scale=layer.w13_scales,
        w2_scale=layer.w2_scales,
        block_shape=old_experts.block_shape,
    ))
    with pytest.raises(RuntimeError, match="already active"):
        worker._start_weight_update(is_draft=is_draft)
    worker.finish_weight_update()
    for scale in captured:
        torch.testing.assert_close(scale, torch.full_like(scale, 3))


def test_v2_reload_refreshes_previous_owner_after_kernel_rebuild(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2

    model, layer, method, captured, executions = _captured_grouped_model(monkeypatch)
    addresses = tuple(scale.data_ptr() for scale in captured)
    generations = iter(range(3, 13))

    def reload(self, *args, **kwargs):
        value = next(generations)
        for name in ("w13_scales", "w2_scales"):
            source = getattr(layer, name)
            setattr(layer, name, torch.nn.Parameter(
                torch.full_like(source, value), requires_grad=False,
            ))
        previous_experts = method.moe_kernel.fused_experts
        new_quant = SimpleNamespace(use_int4_w4a16=True, w1_zp=None, w2_zp=None)
        method.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(
            quant_config=new_quant,
            block_shape=previous_experts.block_shape,
            moe_config=previous_experts.moe_config,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
        ))

    monkeypatch.setattr(GPUModelRunner, "reload_weights", reload)
    monkeypatch.setattr(HcuGPUModelRunnerV2, "get_model", lambda self: model)
    runner = object.__new__(HcuGPUModelRunnerV2)
    for value in range(3, 13):
        runner.reload_weights()
        assert tuple(scale.data_ptr() for scale in captured) == addresses
        for scale in captured:
            torch.testing.assert_close(scale, torch.full_like(scale, value))
        with torch.inference_mode():
            assert _apply_grouped(method.moe_kernel.fused_experts)
        assert executions[-1][0] is captured[0]
        assert executions[-1][1] is captured[1]


@pytest.mark.parametrize("update", ["copy", "data"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.hcu)],
)
def test_checkpoint_restore_then_kernel_update_uses_layer_scales(
    monkeypatch, update, device,
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.model_executor.layers.fused_moe.config import (
        gptq_marlin_moe_quant_config,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        _copy_and_restore_kernel_tensors,
    )
    from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
    from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    model, layer, method, captured, executions = _captured_grouped_model(
        monkeypatch, device,
    )
    graph = None
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            replayed = captured[0].flatten()[0] + captured[1].flatten()[0]
        graph.replay()
        torch.testing.assert_close(replayed, torch.tensor(2.0, device=device))
    original = {name: getattr(layer, name) for name in ("w13_scales", "w2_scales")}
    previous_owners = grouped_w4a16_moe_runtime.snapshot_grouped_w4a16_scale_owners(model)
    for name, source in original.items():
        setattr(layer, name, torch.nn.Parameter(
            torch.full_like(source, 3), requires_grad=False,
        ))

    quant = gptq_marlin_moe_quant_config(
        w1_scale=layer.w13_scales, w2_scale=layer.w2_scales,
        weight_bits=4, group_size=128,
    )

    class ExpertView:
        def __init__(self, quant_config):
            self.quant_config = quant_config
            self.block_shape = [0, 128]
            self.moe_config = SimpleNamespace(moe_backend="aiter")

        @property
        def w1_scale(self):
            return self.quant_config.w1_scale

        @property
        def w2_scale(self):
            return self.quant_config.w2_scale

    experts = ExpertView(quant)
    method.moe_kernel = SimpleNamespace(fused_experts=experts)
    _copy_and_restore_kernel_tensors(
        layer,
        LayerReloadingInfo(
            restore_metadata=({}, {}), restore_device=torch.device(device),
            kernel_tensors=(original, {}),
        ),
    )
    assert layer.w13_scales is original["w13_scales"]
    assert experts.w1_scale is not layer.w13_scales
    grouped_w4a16_moe_runtime.refresh_grouped_w4a16_scale_caches(
        model, previous_owners,
    )
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], 3))

    def load_kernel_format(self, *args, **kwargs):
        for source in (layer.w13_scales, layer.w2_scales):
            with torch.no_grad():
                if update == "data":
                    source.data.copy_(torch.full_like(source, 5))
                else:
                    source.copy_(torch.full_like(source, 5))

    monkeypatch.setattr(GPUModelRunner, "reload_weights", load_kernel_format)
    monkeypatch.setattr(HcuGPUModelRunnerV2, "get_model", lambda self: model)
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.reload_weights()
    for scale in captured:
        torch.testing.assert_close(scale, torch.full_like(scale, 5))
    if graph is not None:
        graph.replay()  # No grouped eager forward after the kernel-format update.
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, torch.tensor(10.0, device=device))
    assert experts.w1_scale is layer.w13_scales
    assert experts.w2_scale is layer.w2_scales
    with torch.inference_mode():
        assert _apply_grouped(experts)
    assert executions[-1][0] is captured[0]
    assert executions[-1][1] is captured[1]


def test_v1_kernel_format_reload_refreshes_captured_grouped_scales(monkeypatch):
    import ast
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/hcu_model_runner.py"
    module = ast.parse(path.read_text())
    runner_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    reload_method = next(
        node for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "reload_weights"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )
    scope = dict(
        torch=torch, time=time,
        logger=SimpleNamespace(
            info_once=lambda *args: None,
            warning=lambda *args: None,
            warning_once=lambda *args: None,
        ),
    )
    extracted = ast.Module(body=[future, reload_method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), scope)

    model, _, _, captured, _ = _captured_grouped_model(monkeypatch)
    addresses = tuple(scale.data_ptr() for scale in captured)
    runner = SimpleNamespace(
        get_model=lambda: model,
        model_config=SimpleNamespace(quantization=None),
    )
    weights = [
        (name, torch.full_like(parameter, 3))
        for name, parameter in model.named_parameters()
    ]
    scope["reload_weights"](
        runner, weights_iterator=weights, is_checkpoint_format=False,
    )

    assert tuple(scale.data_ptr() for scale in captured) == addresses
    for scale in captured:
        torch.testing.assert_close(scale, torch.full_like(scale, 3))


@pytest.mark.hcu
@pytest.mark.parametrize("update", ["copy", "data"])
def test_hip_graph_replays_refreshed_grouped_scales_without_eager(monkeypatch, update):
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    from vllm.v1.worker.gpu_worker import Worker
    from vllm_hcu.v1.worker import HcuGPUWorker

    model, layer, _, captured, _ = _captured_grouped_model(monkeypatch, device="cuda")
    addresses = tuple(scale.data_ptr() for scale in captured)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        replayed = captured[0].flatten()[0] + captured[1].flatten()[0]
    graph.replay()
    torch.testing.assert_close(replayed, torch.tensor(2.0, device="cuda"))

    def write_scales(self):
        for source in (layer.w13_scales, layer.w2_scales):
            with torch.no_grad():
                if update == "data":
                    source.data.copy_(torch.full_like(source, 3))
                else:
                    source.copy_(torch.full_like(source, 3))

    monkeypatch.setattr(Worker, "finish_weight_update", write_scales)
    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(model=model, get_draft_model=lambda: None)
    worker._weight_update_is_draft = False
    worker.finish_weight_update()

    graph.replay()  # No eager grouped forward between update and replay.
    torch.cuda.synchronize()
    assert tuple(scale.data_ptr() for scale in captured) == addresses
    torch.testing.assert_close(replayed, torch.tensor(6.0, device="cuda"))
