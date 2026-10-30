# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU logic checks; LightOp device execution is replaced with an INT32 oracle.

These tests do not import a vLLM runtime or claim HCU kernel/performance coverage.
The optional source contract test checks the adapter against VLLM_SOURCE_ROOT.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
import importlib
import inspect
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config, set_hcu_config
from vllm_hcu.patch.worker.op_opt import patch_lm_head_convrot_int8 as patch


def _stub_vllm_logger(monkeypatch):
    # vllm.logger needs a vLLM install; stdlib logging keeps caplog working.
    import logging

    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)


@pytest.fixture
def runtime(monkeypatch):
    # Only the vLLM type boundary is stubbed. Rotation, post-load conversion,
    # registered storage and the plugin's existing INT8 adapter execute for real.
    base = ModuleType("vllm.model_executor.layers.quantization.base_config")
    base.QuantizeMethodBase = type("QuantizeMethodBase", (), {})
    source_root = os.environ.get("VLLM_SOURCE_ROOT")
    if source_root:
        path = Path(source_root) / "vllm/model_executor/layers/quantization/base_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        contract = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QuantizeMethodBase")
        namespace = {"ABC": ABC, "abstractmethod": abstractmethod, "torch": torch, "nn": nn}
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), contract], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        base.QuantizeMethodBase = namespace["QuantizeMethodBase"]
    monkeypatch.setitem(sys.modules, base.__name__, base)
    _stub_vllm_logger(monkeypatch)
    name = "vllm_hcu.model_executor.layers.quantization.lm_head_convrot_int8"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    # Enforce the real device guard in a separate test. Only these numerical
    # checks are permitted to run the post-load conversion on a CPU tensor.
    monkeypatch.setattr(module, "_require_hcu_backend", lambda _weight: None)
    return module


def _head(n=37, k=512):
    generator = torch.Generator().manual_seed(17)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.randn(n, k, generator=generator).bfloat16(), requires_grad=False)
    layer.weight.weight_loader = object()
    layer.embedding_dim = k
    return layer


def _method(runtime, layer):
    layer.quant_method = runtime.ConvRotINT8LMHeadMethod(SimpleNamespace())
    return layer.quant_method


@pytest.fixture
def int8_oracle(monkeypatch):
    calls = []
    quant = ModuleType("lightop.quant")
    gemm = ModuleType("lightop.gemm_ops")

    def quantize(x):
        scale = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
        return (x.float() / scale).round().clamp(-127, 127).to(torch.int8), scale

    def matmul(a, b, scale_a, scale_b, m, n, k, transpose_flag, out_dtype):
        assert a.dtype == b.dtype == torch.int8
        assert a.is_contiguous() and b.is_contiguous()
        assert scale_a.dtype == scale_b.dtype == torch.float32
        assert transpose_flag == "NT"
        assert a.shape == (m, k) and b.shape == (n, k)
        calls.append((m, n, k))
        accumulator = a.int() @ b.int().T
        return True, (accumulator.float() * scale_a * scale_b.T).to(out_dtype)

    quant.per_token_quant_int8 = quantize
    gemm.hipblaslt_w8a8_gemm = matmul
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "1")
    return calls


def test_sidecar_default_validation_and_spawn_round_trip():
    assert HcuFeatureConfig().lm_head_convrot_int8 is False
    config = SimpleNamespace(additional_config={"other": 1})
    set_hcu_config(config, lm_head_convrot_int8=True)
    restored = pickle.loads(pickle.dumps(config))
    assert get_hcu_config(restored).lm_head_convrot_int8 is True
    assert restored.additional_config["other"] == 1
    assert get_hcu_config(restored).to_dict() != HcuFeatureConfig().to_dict()
    with pytest.raises(TypeError, match="must be bool"):
        HcuFeatureConfig(lm_head_convrot_int8="true")


def test_worker_feature_state_follows_serialized_sidecar():
    from vllm_hcu.patch.worker import _feature_states

    assert _feature_states(None, HcuFeatureConfig())["lm_head_convrot_int8"] is False
    assert _feature_states(None, HcuFeatureConfig(lm_head_convrot_int8=True))["lm_head_convrot_int8"] is True


def test_hadamard_preserves_unquantized_linear_identity(runtime):
    matrix = runtime._hadamard(torch.device("cpu"))
    torch.testing.assert_close(matrix.float() @ matrix.float().T, torch.eye(256), rtol=0, atol=0)
    x = torch.randn(3, 512)
    weight = torch.randn(37, 512)
    rotated = runtime._rotate(x, matrix.float()) @ runtime._rotate(weight, matrix.float()).T
    torch.testing.assert_close(rotated, x @ weight.T, rtol=1e-4, atol=1e-4)


def test_bf16_checkpoint_stays_bf16_until_postload_and_conversion_is_idempotent(runtime, monkeypatch):
    layer = _head()
    original = layer.weight
    checkpoint = layer.weight.detach().clone()
    loader_metadata = original.weight_loader
    method = _method(runtime, layer)
    assert layer.weight.dtype == torch.bfloat16
    monkeypatch.setattr(runtime, "_WEIGHT_CHUNK_ROWS", 7)
    method.process_weights_after_loading(layer)
    assert layer.weight is original and layer.weight.weight_loader is loader_metadata
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (37, 1)
    assert layer.weight_scale.dtype == torch.float32
    assert "_hcu_convrot_hadamard" in dict(layer.named_buffers())
    assert checkpoint.dtype == torch.bfloat16
    pointer = layer.weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == pointer
    assert not any(isinstance(value, torch.Tensor) for value in vars(method).values())


@pytest.mark.parametrize("shape", [(3, 512), (2, 3, 512)])
def test_full_vocab_output_bias_and_noncontiguous_input(runtime, int8_oracle, shape):
    layer = _head()
    reference_weight = layer.weight.detach().float().clone()
    method = _method(runtime, layer)
    method.process_weights_after_loading(layer)
    x = torch.randn(*shape, 2).bfloat16()[..., 0]
    assert not x.is_contiguous()
    bias = torch.randn(37).bfloat16()
    output = method.apply(layer, x, bias)
    assert output.shape == (*shape[:-1], 37)  # Includes the last vocab row.
    assert output.dtype == torch.bfloat16
    reference = torch.nn.functional.linear(x.float(), reference_weight, bias.float())
    relative_error = torch.linalg.vector_norm(output.float() - reference) / torch.linalg.vector_norm(reference)
    assert relative_error < 0.03
    assert int8_oracle == [(x.numel() // 512, 37, 512)]


def test_row_shards_concatenate_to_the_full_head(runtime, int8_oracle):
    full = _head(n=40)
    first, second = _head(n=17), _head(n=23)
    first.weight.data.copy_(full.weight[:17])
    second.weight.data.copy_(full.weight[17:])
    for layer in (full, first, second):
        _method(runtime, layer).process_weights_after_loading(layer)
    x = torch.randn(4, 512).bfloat16()
    full_out = full.quant_method.apply(full, x)
    joined = torch.cat([layer.quant_method.apply(layer, x) for layer in (first, second)], dim=-1)
    torch.testing.assert_close(joined, full_out, rtol=0, atol=0)


def test_zero_weights_and_empty_token_batch(runtime, int8_oracle):
    layer = _head()
    layer.weight.data.zero_()
    method = _method(runtime, layer)
    method.process_weights_after_loading(layer)
    assert torch.isfinite(layer.weight_scale).all()
    assert torch.count_nonzero(layer.weight) == 0
    assert method.apply(layer, torch.empty(0, 512, dtype=torch.bfloat16)).shape == (0, 37)
    assert int8_oracle == []


def test_failed_conversion_does_not_publish_partial_int8_weights(runtime, monkeypatch):
    layer = _head()
    layer.weight.data[-1, 0] = float("nan")
    monkeypatch.setattr(runtime, "_WEIGHT_CHUNK_ROWS", 7)
    with pytest.raises(runtime.ConvRotLMHeadError, match="non-finite"):
        _method(runtime, layer).process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")
    assert not getattr(layer, "_hcu_convrot_ready", False)


def test_device_guard_rejects_cpu_without_loading_lightop(monkeypatch):
    base = ModuleType("vllm.model_executor.layers.quantization.base_config")
    base.QuantizeMethodBase = type("QuantizeMethodBase", (), {})
    monkeypatch.setitem(sys.modules, base.__name__, base)
    _stub_vllm_logger(monkeypatch)
    runtime = importlib.import_module("vllm_hcu.model_executor.layers.quantization.lm_head_convrot_int8")
    with pytest.raises(runtime.ConvRotLMHeadError, match="HCU/HIP"):
        runtime._require_hcu_backend(torch.ones(2, 256, dtype=torch.bfloat16))


def _target_module():
    module = ModuleType(patch.TARGET_MODULE)
    module.DEFAULT_VOCAB_PADDING_SIZE = 64

    class UnquantizedEmbeddingMethod:
        pass

    class ParallelLMHead(nn.Module):
        def __init__(
            self, num_embeddings, embedding_dim, bias=False, params_dtype=None,
            org_num_embeddings=None, padding_size=64, quant_config=None, prefix="",
            *, disable_tp=False,
        ):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.num_embeddings_per_partition = num_embeddings
            self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, dtype=params_dtype or torch.bfloat16), requires_grad=False)
            self.quant_method = UnquantizedEmbeddingMethod()

    module.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
    module.ParallelLMHead = ParallelLMHead
    return module


def test_adapter_default_off_and_enabled_postload_selection(runtime, monkeypatch):
    module = _target_module()
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    config_module = ModuleType("vllm.config")
    config = SimpleNamespace(additional_config={})
    config_module.get_current_vllm_config_or_none = lambda: config
    for item in (module, linear, config_module):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    off = module.ParallelLMHead(37, 512)
    assert type(off.quant_method) is module.UnquantizedEmbeddingMethod
    set_hcu_config(config, lm_head_convrot_int8=True)
    on = module.ParallelLMHead(37, 512)
    assert isinstance(on.quant_method, runtime.ConvRotINT8LMHeadMethod)
    assert on.weight.dtype == torch.bfloat16
    with pytest.raises(runtime.ConvRotLMHeadError, match="finalized"):
        on.quant_method.apply(on, torch.ones(2, 512, dtype=torch.bfloat16))
    with pytest.raises(runtime.ConvRotLMHeadError, match="tied"):
        on.quant_method.tie_weights(on, off)


def test_default_off_does_not_import_convrot_runtime(monkeypatch):
    import builtins

    module = _target_module()
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config_or_none = lambda: None
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.endswith(".lm_head_convrot_int8")
        assert not name.startswith("lightop")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    patch.apply_to_module(module)
    layer = module.ParallelLMHead(37, 512)
    assert type(layer.quant_method) is module.UnquantizedEmbeddingMethod


@pytest.mark.parametrize("vocab_size", [768, 512])
def test_nn_layout_linear_head_is_skipped_before_conversion(runtime, monkeypatch, caplog, vocab_size):
    module = _target_module()
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    monkeypatch.delenv("VLLM_USE_NN", raising=False)  # Default is enabled.
    layer = module.ParallelLMHead(vocab_size, 512)
    layer.weight = nn.Parameter(torch.zeros(512, vocab_size, dtype=torch.bfloat16), requires_grad=False)
    layer.quant_method = linear.UnquantizedLinearMethod()
    original_weight, original_method = layer.weight, layer.quant_method

    assert runtime.install_convrot_lm_head(layer, SimpleNamespace()) is False
    assert "VLLM_USE_NN" in caplog.text
    assert layer.weight is original_weight
    assert layer.quant_method is original_method


def test_output_major_linear_head_is_supported_without_nn_layout(runtime, monkeypatch, int8_oracle):
    module = _target_module()
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    monkeypatch.setenv("VLLM_USE_NN", "0")
    layer = module.ParallelLMHead(37, 512)
    layer.quant_method = linear.UnquantizedLinearMethod()

    runtime.install_convrot_lm_head(layer, SimpleNamespace())
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.quant_method.apply(layer, torch.ones(2, 512, dtype=torch.bfloat16)).shape == (2, 37)
    assert int8_oracle == [(2, 37, 512)]


@pytest.mark.parametrize("shape", [(512, 37), (38, 512), (37 * 512,)])
def test_skips_weight_shape_inconsistent_with_head_metadata(runtime, monkeypatch, caplog, shape):
    module = _target_module()
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    layer = module.ParallelLMHead(37, 512)
    layer.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
    original_method = layer.quant_method

    assert runtime.install_convrot_lm_head(layer, SimpleNamespace()) is False
    assert "weight shape" in caplog.text
    assert layer.quant_method is original_method


@pytest.mark.parametrize("policy", ["quantized", "fp16", "tied", "fp32_head", "lora", "bad_k"])
def test_enabled_head_skips_incompatible_policies(runtime, monkeypatch, caplog, policy):
    module = _target_module()
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    layer = module.ParallelLMHead(37, 512)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))
    if policy == "quantized":
        layer.quant_method = object()
    elif policy == "fp16":
        layer.weight.data = layer.weight.data.half()
    elif policy == "tied":
        config.model_config.hf_text_config.tie_word_embeddings = True
    elif policy == "fp32_head":
        config.model_config.head_dtype = torch.float32
    elif policy == "lora":
        config.lora_config = object()
    elif policy == "bad_k":
        layer.embedding_dim = 300
    original_method = layer.quant_method
    assert runtime.install_convrot_lm_head(layer, config) is False
    assert "keeping BF16" in caplog.text
    assert layer.quant_method is original_method


def _real_runtime(monkeypatch):
    # The `runtime` fixture stubs the backend guard; load an unstubbed copy.
    base = ModuleType("vllm.model_executor.layers.quantization.base_config")
    base.QuantizeMethodBase = type("QuantizeMethodBase", (), {})
    monkeypatch.setitem(sys.modules, base.__name__, base)
    _stub_vllm_logger(monkeypatch)
    name = "vllm_hcu.model_executor.layers.quantization.lm_head_convrot_int8"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_backend_guard_leaves_lightop_availability_to_the_int8_runtime(monkeypatch):
    runtime = _real_runtime(monkeypatch)
    # apply_int8_linear already reports a missing LightOp; the guard only checks
    # the device and the custom quantization GEMM switch.
    monkeypatch.setitem(sys.modules, "lightop", None)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(
        importlib.import_module("vllm_hcu.platforms.envs"),
        "custom_quantization_gemm_enabled", lambda: True,
    )
    runtime._require_hcu_backend(SimpleNamespace(is_cuda=True))


def test_backend_guard_requires_custom_quantization_gemm(monkeypatch):
    runtime = _real_runtime(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(
        importlib.import_module("vllm_hcu.platforms.envs"),
        "custom_quantization_gemm_enabled", lambda: False,
    )
    with pytest.raises(runtime.ConvRotLMHeadError, match="CUSTOM_QUANTIZATION_GEMM"):
        runtime._require_hcu_backend(SimpleNamespace(is_cuda=True))


def _worker_guard():
    # vllm_hcu.v1.worker needs a full vLLM; execute only the guard functions.
    source = (Path(__file__).parents[2] / "vllm_hcu/v1/worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_has_convrot_lm_head", "_reject_convrot_checkpoint_reload"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(nodes, []), "worker", "exec"), namespace)
    return tree, namespace["_reject_convrot_checkpoint_reload"]


def test_checkpoint_reload_guard_rejects_only_checkpoint_format_on_convrot_head(runtime):
    _, reject = _worker_guard()
    plain = nn.Sequential(nn.Linear(2, 2), _head())
    convrot = nn.Sequential(nn.Linear(2, 2), _head())
    convrot[1].quant_method = runtime.ConvRotINT8LMHeadMethod(SimpleNamespace())

    reject(plain, (), {})  # no ConvRot head: never blocked
    reject(plain, (None, None, True), {})
    for args, kwargs in (((), {}), ((None, None, True), {}), ((), {"is_checkpoint_format": True})):
        with pytest.raises(RuntimeError, match="checkpoint-format"):
            reject(convrot, args, kwargs)
    reject(convrot, (None, None, False), {})  # kernel-format reload is allowed
    reject(convrot, (), {"is_checkpoint_format": False})


def test_worker_reload_weights_runs_the_guard_before_the_runner():
    tree, _ = _worker_guard()
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HcuGPUWorker")
    method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "reload_weights")
    calls = [
        ast.unparse(n.value.func) for n in method.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    ]
    assert calls.index("_reject_convrot_checkpoint_reload") < calls.index("super().reload_weights")


def test_adapter_constructor_contract_matches_target_source():
    source_root = os.environ.get("VLLM_SOURCE_ROOT")
    if not source_root:
        pytest.skip("set VLLM_SOURCE_ROOT for the v0.28.1 source contract check")
    path = Path(source_root) / "vllm/model_executor/layers/vocab_parallel_embedding.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    head = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ParallelLMHead")
    constructor = next(node for node in head.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    # Compile only the signature; the actual source supplies names and defaults.
    constructor.body = [ast.Pass()]
    constructor.decorator_list = []
    namespace = {"DEFAULT_VOCAB_PADDING_SIZE": 64}
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), constructor], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    module = _target_module()
    assert inspect.signature(namespace["__init__"]).parameters.keys() == inspect.signature(module.ParallelLMHead.__init__).parameters.keys()
    module.ParallelLMHead.__init__ = namespace["__init__"]
    assert patch.apply_to_module(module) is True
