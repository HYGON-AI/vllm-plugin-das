# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.utils import weights_already_processed
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterInt8ScaledMMLinearKernel

from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8LinearMethod,
)


def test_aiter_processed_weight_is_not_transposed_twice():
    kernel = object.__new__(AiterInt8ScaledMMLinearKernel)
    kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
    kernel.config = SimpleNamespace(is_channelwise=True, is_static_input_scheme=False,
                                    input_symmetric=True)
    class Scheme:
        def create_weights(self, **kwargs):
            self.kernel = kernel
        def process_weights_after_loading(self, layer):
            self.kernel.process_weights_after_loading(layer)

    layer = torch.nn.Module()
    layer.scheme = Scheme()
    layer.logical_widths = [2]
    original = torch.tensor([[1, 2, 3], [-2, 4, 5]], dtype=torch.int8)
    layer.weight = torch.nn.Parameter(original, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.tensor([[0.5], [0.25]]), requires_grad=False)
    method = SlimQuantW4A8Int8LinearMethod(None)
    method.create_weights(layer, 3, [2], 3, 2, torch.float32)
    assert method.supports_pre_processed_weights
    method.process_weights_after_loading(layer)
    processed = layer.weight.detach().clone()
    assert torch.equal(processed, original.T)
    with weights_already_processed():
        method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight, processed)
    x = torch.tensor([[2., -1., 3.]])
    assert torch.equal((x @ layer.weight.float()) * layer.weight_scale.T,
                       (x @ original.float().T) * layer.weight_scale.T)


def test_unverified_kernel_rejects_preprocessed():
    layer = SimpleNamespace(scheme=SimpleNamespace(create_weights=lambda **kw: None))
    method = SlimQuantW4A8Int8LinearMethod(None)
    method.create_weights(layer, 3, [2], 3, 2, torch.float32)
    assert not method.supports_pre_processed_weights
    with weights_already_processed(), pytest.raises(RuntimeError, match="cannot restore"):
        method.process_weights_after_loading(layer)


def test_ipc_loader_reconstructs_processed_linear_without_repacking(monkeypatch):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries
    from vllm.model_executor.model_loader.weight_cache import ipc_loader
    from vllm.model_executor.utils import is_weights_pre_processed

    class Scheme:
        def create_weights(self, **kwargs):
            self.kernel = object.__new__(AiterInt8ScaledMMLinearKernel)
        def process_weights_after_loading(self, layer):
            pytest.fail("IPC reconstruction must not transform processed weights")

    def make_model(**kwargs):
        layer = torch.nn.Module()
        layer.scheme = Scheme()
        layer.weight = torch.nn.Parameter(torch.empty((2, 3), dtype=torch.int8), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.empty((2, 1)), requires_grad=False)
        layer.quant_method = SlimQuantW4A8Int8LinearMethod(None)
        layer.quant_method.create_weights(layer, 3, [2], 3, 2, torch.float32)
        return layer

    def post_load(model, model_config, target_device):
        assert is_weights_pre_processed()
        model.quant_method.process_weights_after_loading(model)

    monkeypatch.setattr(ipc_loader, "initialize_model", make_model)
    monkeypatch.setattr(ipc_loader, "process_weights_after_loading", post_load)
    cached = torch.nn.Module()
    cached.weight = torch.nn.Parameter(torch.tensor([[1, -2], [3, 4], [5, 6]], dtype=torch.int8), requires_grad=False)
    cached.weight_scale = torch.nn.Parameter(torch.tensor([[0.25], [0.5]]), requires_grad=False)
    entries, aliases = export_entries(cached)
    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"fallback": False}))
    config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu:0")))
    model_config = SimpleNamespace(dtype=torch.float32)
    result = loader._build_model(config, model_config, "", entries, aliases)
    assert torch.equal(result.weight, cached.weight)
    assert torch.equal(result.weight_scale, cached.weight_scale)
    assert result.weight.shape == (3, 2)


def test_ipc_loader_rejects_missing_meta_parameter(monkeypatch):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader
    from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry
    from vllm_hcu.runtime_compat.weight_cache import install_loader_compat

    cls = ipc_loader.IpcModelLoader
    for name in ("_fetch_entries", "_resolve_socket_path", "_check_supported", "_apply_entries"):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    monkeypatch.setattr(cls, "_hcu_weight_cache_loader", False, raising=False)
    monkeypatch.setattr(ipc_loader, "_materialize_remaining_meta_tensors",
                        ipc_loader._materialize_remaining_meta_tensors)
    install_loader_compat(ipc_loader)
    monkeypatch.setattr(ipc_loader, "initialize_model", lambda **kwargs: torch.nn.Linear(3, 2))
    monkeypatch.setattr(ipc_loader, "process_weights_after_loading", lambda *args: None)
    loader = cls(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"fallback": False}))
    config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu:0")))
    weights = {"weight": TensorEntry.from_tensor(torch.ones(2, 3), "param")}
    with pytest.raises(RuntimeError, match="missing model tensors: bias"):
        loader._build_model(config, SimpleNamespace(dtype=torch.float32), "", weights, {})


def test_hcu_unquantized_preprocessed_support_is_device_scoped():
    # Importing both official and HCU replacement linear modules in one
    # process duplicates custom-op registrations. Extract this property in
    # isolation so the test can exercise its real implementation safely.
    import ast
    from pathlib import Path
    source = Path(__file__).parents[2] / "vllm_hcu/model_executor/layers/linear.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "UnquantizedLinearMethod")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "supports_pre_processed_weights")
    function = ast.Module(body=[method], type_ignores=[])
    namespace = {"current_platform": SimpleNamespace(is_cuda_alike=lambda: True)}
    exec(compile(function, str(source), "exec"), namespace)
    prop = namespace["supports_pre_processed_weights"]
    assert prop.fget(None)
    namespace["current_platform"] = SimpleNamespace(is_cuda_alike=lambda: False)
    assert not prop.fget(None)


def test_triton_w8a8_weights_are_not_reprocessed():
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
    class Scheme:
        def create_weights(self, **kwargs):
            self.kernel = object.__new__(TritonInt8ScaledMMLinearKernel)
        def process_weights_after_loading(self, layer):
            pytest.fail("cached Triton tensor must not be repacked")
    layer = SimpleNamespace(scheme=Scheme())
    method = SlimQuantW4A8Int8LinearMethod(None)
    method.create_weights(layer, 3, [2], 3, 2, torch.float32)
    assert method.supports_pre_processed_weights
    with weights_already_processed():
        method.process_weights_after_loading(layer)
