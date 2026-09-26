# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests for the Qwen4Exp PLE INT8 UVA CPU-offload path."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm

from vllm.config import set_current_vllm_config

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_ple_int8 as patch


TARGET_VLLM_ROOT = Path(
    os.environ.get(
        "VLLM_SOURCE_ROOT",
        Path(vllm.__file__).resolve().parents[1],
    )
).resolve()


def _target_class(relative_path, class_name):
    source = TARGET_VLLM_ROOT / relative_path
    if not source.is_file():
        pytest.skip(f"target vLLM source is unavailable: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def test_target_vllm_exposes_engram_cpu_offload_cli_contract():
    engram = _target_class("vllm/config/engram.py", "EngramConfig")
    assert any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "cpu_offload"
        for node in engram.body
    )

    engine_args = _target_class("vllm/engine/arg_utils.py", "EngineArgs")
    assert any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "engram_config"
        for node in engine_args.body
    )
    add_cli_args = next(
        node
        for node in engine_args.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "add_cli_args"
    )
    assert any(
        isinstance(node, ast.Constant) and node.value == "--engram-config"
        for node in ast.walk(add_cli_args)
    )


def _make_table(rows: int, dim: int):
    torch.manual_seed(0)
    weight = torch.randint(-128, 128, (rows, dim), dtype=torch.int8)
    weight_scale = torch.rand(rows, 1, dtype=torch.bfloat16) + 0.1
    return weight, weight_scale


def _reference(weight, weight_scale, ids, dtype):
    embeddings = torch.nn.functional.embedding(ids, weight)
    scales = torch.nn.functional.embedding(ids, weight_scale)
    return embeddings.to(dtype) * scales.to(dtype)


def _int8_quant_config(prefix):
    weight_quant = SimpleNamespace(type="int", num_bits=8, strategy="channel")
    return SimpleNamespace(
        get_name=lambda: "compressed-tensors",
        target_scheme_map={prefix: {"weights": weight_quant}},
    )


def _storage_class(quant_config):
    class BaseEmbedding:
        def __init__(self, *args, **kwargs):
            del args
            self.quant_method = kwargs.get("quant_method")

    module = SimpleNamespace(PLEVocabParallelEmbedding=BaseEmbedding)
    return patch._make_storage_class(module, quant_config)


def test_slimquant_unquantized_ple_honors_cpu_offload(monkeypatch):
    # The BF16 PLE table must not consume ~24 GiB per rank on TP4.
    from vllm.model_executor import parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(patch, "_should_offload_ple_to_cpu", lambda: True)
    monkeypatch.setattr(patch, "_is_uva_available", lambda: True)
    monkeypatch.setattr(patch, "_is_pin_memory_available", lambda: False)
    config = SimpleNamespace(get_name=lambda: "slimquant_w4a8")
    storage = _storage_class(config)(prefix="model.layers.2.ple.ngram_embedding")
    method = storage.quant_method
    assert method is not None, "SlimQuant PLE did not select CPU offload"
    layer = torch.nn.Module()
    method.create_weights(layer, 3, [4], 3, 4, torch.bfloat16,
                          weight_loader=lambda *args: None)
    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.bfloat16
    assert tuple(layer.weight.shape) == (4, 3)
    assert not hasattr(layer, "weight_scale")
    assert method.requires_device_loading is False


def test_slimquant_ple_offload_requires_uva(monkeypatch):
    monkeypatch.setattr(patch, "_should_offload_ple_to_cpu", lambda: True)
    monkeypatch.setattr(patch, "_is_uva_available", lambda: False)
    config = SimpleNamespace(get_name=lambda: "slimquant_w4a8")
    with pytest.raises(RuntimeError, match="UVA"):
        _storage_class(config)(prefix="model.layers.2.ple.ngram_embedding")


def test_slimquant_ple_offload_disabled_preserves_default(monkeypatch):
    monkeypatch.setattr(patch, "_should_offload_ple_to_cpu", lambda: False)
    config = SimpleNamespace(get_name=lambda: "slimquant_w4a8")
    storage = _storage_class(config)(prefix="model.layers.2.ple.ngram_embedding")
    assert storage.quant_method is None


def test_unquantized_ple_uva_lookup_prefetch_and_reload(monkeypatch):
    method = patch.HcuQwen4ExpPLEUnquantizedUVAEmbeddingMethod()
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.tp_size = 1
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16),
        requires_grad=False,
    )
    monkeypatch.setattr(torch.ops._C, "get_cuda_view_from_cpu_tensor",
                        lambda x: x, raising=False)
    ids = torch.tensor([[2, 0]])
    expected = torch.tensor([[[5, 6], [1, 2]]], dtype=torch.bfloat16)
    torch.testing.assert_close(method.embedding(layer, ids), expected)
    output = torch.empty_like(expected)
    method.prepare_prefetch(layer)
    assert method.is_prefetch_prepared(layer)
    method.prefetch_lookup_into(layer, ids, output)
    torch.testing.assert_close(method.finalize_prefetched(layer, output), expected)
    monkeypatch.setattr(patch, "tensor_model_parallel_all_reduce", lambda x: x * 2)
    layer.tp_size = 2
    torch.testing.assert_close(method.finalize_prefetched(layer, output), expected * 2)
    layer.weight = torch.nn.Parameter(layer.weight.detach().clone() + 10,
                                      requires_grad=False)
    method.process_weights_after_loading(layer)
    assert not method.is_prefetch_prepared(layer)
    torch.testing.assert_close(method.embedding(layer, ids), expected + 10)


@pytest.mark.parametrize("unquantized", [False, True])
def test_uva_post_load_keeps_parameter_storage_on_cpu(
    monkeypatch: pytest.MonkeyPatch, unquantized,
):
    from vllm.model_executor.model_loader import utils as loader_utils

    layer = torch.nn.Module()
    layer.quant_method = (
        patch.HcuQwen4ExpPLEUnquantizedUVAEmbeddingMethod() if unquantized
        else patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
    )
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty((4, 3), dtype=torch.int8), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.empty((4, 1), dtype=torch.bfloat16), requires_grad=False
        ),
    )
    model = torch.nn.Module()
    model.embedding = layer
    weight_ptr = layer.weight.data_ptr()
    scale_ptr = layer.weight_scale.data_ptr()
    staged_modules = []

    def track_staging(module, target_device):
        staged_modules.append((module, target_device))
        return pytest.fail("CPU-offloaded PLE must not enter device staging")

    monkeypatch.setattr(loader_utils, "device_loading_context", track_staging)
    monkeypatch.setattr(loader_utils, "maybe_retie_word_embeddings", lambda *args: None)
    monkeypatch.setattr(
        loader_utils, "release_device_memory_under_pressure", lambda *args: None
    )

    loader_utils.process_weights_after_loading(
        model,
        SimpleNamespace(quantization=None),
        torch.device("cuda"),
    )

    assert layer.quant_method.requires_device_loading is False
    assert layer.weight.device.type == "cpu"
    assert layer.weight_scale.device.type == "cpu"
    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight_scale.data_ptr() == scale_ptr
    assert staged_modules == []


def test_offload_toggle_falls_back_to_hcu_environment(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_PLE_CPU_OFFLOAD", raising=False)
    assert patch._should_offload_ple_to_cpu() is False
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")
    assert patch._should_offload_ple_to_cpu() is True
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "0")
    assert patch._should_offload_ple_to_cpu() is False


@pytest.mark.parametrize(
    ("configured", "legacy"),
    ((False, "1"), (True, "0")),
)
def test_explicit_engram_config_takes_precedence(
    monkeypatch, configured, legacy
):
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", legacy)
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=configured)
    )

    with set_current_vllm_config(vllm_config):
        assert patch._should_offload_ple_to_cpu() is configured


def test_missing_engram_config_uses_hcu_environment(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")

    with set_current_vllm_config(SimpleNamespace(engram_config=None)):
        assert patch._should_offload_ple_to_cpu() is True


def test_explicit_disabled_engram_config_uses_resident_method(monkeypatch):
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    storage_class = _storage_class(_int8_quant_config(prefix))
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setattr(patch, "_is_uva_available", lambda: False)
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=False),
        quant_config=None,
    )

    with set_current_vllm_config(vllm_config):
        embedding = storage_class(prefix=prefix, params_dtype=torch.bfloat16)

    assert isinstance(
        embedding.quant_method, patch.HcuQwen4ExpPLEInt8EmbeddingMethod
    )


def test_cpu_offload_requires_uva_operator(monkeypatch):
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    storage_class = _storage_class(_int8_quant_config(prefix))
    monkeypatch.setattr(patch, "_is_uva_available", lambda: False)
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=True),
        quant_config=None,
    )

    with set_current_vllm_config(vllm_config):
        with pytest.raises(RuntimeError, match="CPU offload requires the HCU UVA"):
            storage_class(prefix=prefix, params_dtype=torch.bfloat16)


def test_cpu_offload_selects_uva_method(monkeypatch):
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    storage_class = _storage_class(_int8_quant_config(prefix))
    monkeypatch.setattr(patch, "_is_uva_available", lambda: True)
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=True),
        quant_config=None,
    )

    with set_current_vllm_config(vllm_config):
        embedding = storage_class(prefix=prefix, params_dtype=torch.bfloat16)

    assert isinstance(
        embedding.quant_method, patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod
    )


def test_uva_create_weights_allocates_on_cpu():
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        if ps._TP is None:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29591",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)

        method = patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[10],
            input_size=16,
            output_size=10,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )

    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight.shape) == (10, 16)
    assert layer.weight_scale.device.type == "cpu"
    assert layer.weight_scale.dtype == torch.bfloat16
    assert tuple(layer.weight_scale.shape) == (10, 1)


# --- UVA zero-copy variant ---------------------------------------------------

_UVA_AVAILABLE = patch._is_uva_available()
_needs_uva = pytest.mark.skipif(
    not _UVA_AVAILABLE, reason="UVA op get_cuda_view_from_cpu_tensor unavailable"
)
_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires accelerator"
)


def test_uva_method_is_subclass_for_loader_isinstance():
    # The ngram loader routes INT8 scale shards by isinstance against the
    # resident method, so the UVA method must remain its subclass.
    assert issubclass(
        patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod,
        patch.HcuQwen4ExpPLEInt8EmbeddingMethod,
    )


@_needs_uva
@_needs_cuda
def test_uva_method_embedding_matches_resident():
    # UVA lookup must be numerically identical to the GPU-resident dequant.
    weight, weight_scale = _make_table(48, 12)
    w_pin = weight.pin_memory()
    s_pin = weight_scale.pin_memory()
    ids = torch.tensor([2, 2, 47, 0], device="cuda", dtype=torch.long)

    uva = patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=w_pin),
        weight_scale=SimpleNamespace(data=s_pin),
        params_dtype=torch.bfloat16,
        _hcu_uva_views=None,
    )
    got = uva.embedding(layer, ids)
    want = _reference(weight, weight_scale, ids.cpu(), torch.bfloat16)

    assert got.shape == (4, 12)
    assert torch.equal(got.cpu(), want)


@_needs_uva
@_needs_cuda
def test_uva_view_is_cached_across_calls():
    # The device view is created once and reused; a second lookup must not
    # rebuild it (rebuilding every step would defeat the zero-copy path).
    weight, weight_scale = _make_table(32, 8)
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=weight.pin_memory()),
        weight_scale=SimpleNamespace(data=weight_scale.pin_memory()),
        params_dtype=torch.bfloat16,
        _hcu_uva_views=None,
    )
    uva = patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()

    ids = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    uva.embedding(layer, ids)
    first = layer._hcu_uva_views
    uva.embedding(layer, ids)
    second = layer._hcu_uva_views

    assert first is not None
    assert first is second
