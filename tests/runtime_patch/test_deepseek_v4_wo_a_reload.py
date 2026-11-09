# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reload must update the wo_a tensor already referenced by captured graphs."""

from __future__ import annotations

import ast
import importlib.util
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_wo_a_layout as patch


ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = Path(importlib.util.find_spec("vllm").origin).parent


def _function(path, name, class_name=None, namespace=None):
    # Exercise the real getter/reload bodies without constructing a GPU runner
    # or importing the retained V1 runner's unrelated legacy dependencies.
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name)
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(namespace or {}, torch=torch)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


@pytest.fixture
def getter():
    original = _function(
        VLLM_ROOT / "v1/attention/ops/rocm_aiter_mla_sparse.py",
        "_get_cached_wo_a_bf16",
    )
    module = ModuleType(patch.TARGET_MODULE)
    module._get_cached_wo_a_bf16 = original
    patch.apply_to_module(module)
    return module._get_cached_wo_a_bf16


class _Model(torch.nn.Module):
    def __init__(self, logical_weight, use_nn):
        super().__init__()
        self.wo_a = torch.nn.Module()
        self.wo_a.is_quantization = False
        self.use_nn = use_nn
        physical = logical_weight.T.contiguous() if use_nn else logical_weight.clone()
        self.wo_a.weight = torch.nn.Parameter(physical, requires_grad=False)

    def load_weights(self, weights):
        loaded = set()
        for name, weight in weights:
            self.get_parameter(name).copy_(weight.T if self.use_nn else weight)
            loaded.add(name)
        return loaded


def _runner(kind, model, monkeypatch):
    log = SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None,
                          warning=lambda *a: None)
    scope = dict(time=time, logger=log,
                 initialize_layerwise_reload=lambda model: None,
                 finalize_layerwise_reload=lambda model, config: None,
                 _get_parameter_for_reload=lambda model, name: model.get_parameter(name))
    if kind == "v1":
        reload = _function(ROOT / "vllm_hcu/v1/hcu_model_runner.py", "reload_weights",
                           "GPUModelRunner", scope)
        runner = SimpleNamespace(model=model, get_model=lambda: model)
    else:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2

        # V2 delegates to this native V1 reload implementation. Replace only
        # that delegation boundary; the HCU V2 override itself stays real.
        native_reload = _function(VLLM_ROOT / "v1/worker/gpu_model_runner.py",
                                  "reload_weights", "GPUModelRunner", scope)
        monkeypatch.setattr(GPUModelRunner, "reload_weights", native_reload)
        runner = object.__new__(HcuGPUModelRunnerV2)
        runner.model = model
        runner.reset_lora_state = lambda: None
        runner.reset_encoder_cache = lambda: None
        runner.reset_mm_cache = lambda: None
        runner.lora_config = None
        reload = HcuGPUModelRunnerV2.reload_weights
    runner.model_config = SimpleNamespace(quantization=None)
    return runner, reload


@pytest.mark.parametrize("runner_kind", ["v1", "v2"])
@pytest.mark.parametrize("use_nn", [False, True])
@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_reload_refreshes_existing_wo_a_projection(
    monkeypatch, getter, runner_kind, use_nn, square, checkpoint
):
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_USE_NN", use_nn)
    groups, rank, hidden = (4 if square else 2), 2, 8
    logical = (torch.arange(groups * rank * hidden).reshape(groups * rank, hidden)
               .remainder(7).sub(3).to(torch.bfloat16))
    model = _Model(logical, use_nn)
    x = torch.ones(2, groups, hidden, dtype=torch.bfloat16)
    with torch.inference_mode():
        cached = getter(model.wo_a, groups, rank, hidden)
        first = torch.einsum("tgd,grd->tgr", x, cached).clone()
    pointer = cached.data_ptr()
    updated = logical + 1
    physical = updated.T.contiguous() if use_nn else updated
    runner, reload = _runner(runner_kind, model, monkeypatch)
    with torch.inference_mode():
        reload(runner, weights_iterator=iter([("wo_a.weight", updated if checkpoint
                                              else physical)]),
               is_checkpoint_format=checkpoint)
        after = getter(model.wo_a, groups, rank, hidden)
        actual = torch.einsum("tgd,grd->tgr", x, after)
    torch.testing.assert_close(model.wo_a.weight, physical)
    assert after is cached
    assert after.data_ptr() == pointer
    torch.testing.assert_close(after, updated.reshape(groups, rank, hidden))
    torch.testing.assert_close(actual, first + hidden)


@pytest.mark.hcu
@pytest.mark.parametrize("runner_kind", ["v1", "v2"])
@pytest.mark.parametrize("square", [False, True])
def test_reload_updates_previously_captured_wo_a_projection(
    monkeypatch, getter, runner_kind, square
):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU for graph replay")
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_USE_NN", True)
    groups, rank, hidden = (4 if square else 2), 2, 8
    logical = torch.zeros(groups * rank, hidden, dtype=torch.bfloat16, device="cuda")
    model = _Model(logical, True)
    x = torch.ones(2, groups, hidden, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        cached = getter(model.wo_a, groups, rank, hidden)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.einsum("tgd,grd->tgr", x, cached)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = torch.einsum("tgd,grd->tgr", x, cached)
    runner, reload = _runner(runner_kind, model, monkeypatch)
    pointer = cached.data_ptr()
    # Refresh is deliberately called outside the forward's inference context.
    reload(runner, weights_iterator=iter([("wo_a.weight", torch.ones_like(logical.T))]),
           is_checkpoint_format=False)
    assert model.wo_a._dsv4_wo_a_bf16 is cached
    assert cached.data_ptr() == pointer
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, torch.full_like(captured, hidden))


@pytest.mark.parametrize("draft", [False, True])
def test_finish_weight_update_refreshes_only_selected_model(monkeypatch, getter, draft):
    from vllm.v1.worker.gpu_worker import Worker
    from vllm_hcu.v1.worker import HcuGPUWorker

    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_USE_NN", True)
    target = _Model(torch.zeros(8, 8, dtype=torch.bfloat16), True)
    draft_model = _Model(torch.zeros(8, 8, dtype=torch.bfloat16), True)
    selected = draft_model if draft else target
    with torch.inference_mode():
        caches = [getter(model.wo_a, 4, 2, 8) for model in (target, draft_model)]
    pointers = [cache.data_ptr() for cache in caches]
    monkeypatch.setattr(Worker, "finish_weight_update",
                        lambda self: selected.wo_a.weight.add_(1))
    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(model=target, get_draft_model=lambda: draft_model)
    worker._weight_update_is_draft = draft
    worker.finish_weight_update()
    assert [cache.data_ptr() for cache in caches] == pointers
    for model, cache in zip((target, draft_model), caches):
        expected = 1 if model is selected else 0
        torch.testing.assert_close(cache, torch.full_like(cache, expected))


def test_refresh_keeps_cold_layers_lazy_and_handles_replaced_parameter(monkeypatch, getter):
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_USE_NN", True)
    model = _Model(torch.zeros(8, 8, dtype=torch.bfloat16), True)
    patch.refresh_wo_a_bf16_caches(None)
    patch.refresh_wo_a_bf16_caches(model)
    assert not hasattr(model.wo_a, "_dsv4_wo_a_bf16")
    with torch.inference_mode():
        cached = getter(model.wo_a, 4, 2, 8)
    pointer = cached.data_ptr()
    new_weight = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    model.wo_a.weight = torch.nn.Parameter(new_weight.T.contiguous(), requires_grad=False)
    patch.refresh_wo_a_bf16_caches(model)
    assert model.wo_a._dsv4_wo_a_bf16 is cached
    assert cached.data_ptr() == pointer
    torch.testing.assert_close(cached, new_weight.reshape(4, 2, 8))


def test_refresh_rejects_incompatible_reloaded_layout(monkeypatch, getter):
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_USE_NN", True)
    model = _Model(torch.zeros(8, 8, dtype=torch.bfloat16), True)
    with torch.inference_mode():
        cached = getter(model.wo_a, 4, 2, 8)
    model.wo_a.weight = torch.nn.Parameter(torch.zeros(4, 16), requires_grad=False)
    with pytest.raises(RuntimeError, match="incompatible cache layout"):
        patch.refresh_wo_a_bf16_caches(model)
    assert model.wo_a._dsv4_wo_a_bf16 is cached
