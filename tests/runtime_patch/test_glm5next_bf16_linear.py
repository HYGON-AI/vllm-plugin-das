# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""GLM5Next BF16 NN projection accuracy and optional-op boundaries."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_glm5next_channel_fp8 as patch
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.platforms import hcu


def _model_module(weight, method_factory=None):
    from vllm_hcu.model_executor.layers.linear import UnquantizedLinearMethod

    method_factory = method_factory or UnquantizedLinearMethod
    module = ModuleType(patch.TARGET_MODULE)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.quant_method = method_factory()

    class Glm5NextDecoderLayer(torch.nn.Module):
        def __init__(
            self,
            vllm_config,
            config,
            layer_idx,
            prefix="",
            topk_indices_buffer=None,
            is_mtp_layer=False,
            **kwargs,
        ):
            super().__init__()
            self.is_mtp_layer = is_mtp_layer
            self.proj = Projection()

    class Glm5NextForConditionalGeneration:
        def __init__(self, *, vllm_config, prefix=""):
            pass

    def _try_load_fp8_attn_proj(
        name,
        tensor,
        buf,
        params_dict,
        loaded_params,
        kv_a_pad_size,
    ):
        return False

    module.Glm5NextDecoderLayer = Glm5NextDecoderLayer
    module.Glm5NextForConditionalGeneration = Glm5NextForConditionalGeneration
    module._try_load_fp8_attn_proj = _try_load_fp8_attn_proj
    return module


@pytest.fixture
def optional_ops(monkeypatch):
    from vllm import envs
    from vllm.model_executor.custom_op import PluggableLayer

    # Load the real HCU method without registering replacement linear classes
    # in pytest's shared upstream custom-op registry. Restore the module cache
    # after each test so neighboring runtime-patch tests retain their imports.
    name = "vllm_hcu.model_executor.layers.linear"
    path = (
        Path(__file__).resolve().parents[2] / "vllm_hcu/model_executor/layers/linear.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    linear = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, linear)
    with monkeypatch.context() as registration:
        registration.setattr(PluggableLayer, "register", lambda name: lambda cls: cls)
        spec.loader.exec_module(linear)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM53_GATE_UP_DEEPGEMM", "0")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(hcu, "on_gfx938", lambda: True)
    for name in (patch.KPOOL_MODULE, patch.ATTENTION_MODULE, patch.MTP_MODULE):
        monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize("rows", [119, 120, 121, 127, 128])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not hcu.on_gfx938(),
    reason="requires the gfx938 BF16 NN GEMM backend",
)
def test_glm5next_bf16_nn_matches_fp32_reference(optional_ops, monkeypatch, rows):
    # Public deterministic operands reproduce the first-layer projection shape;
    # this test contains no checkpoint weights or captured model activations.
    generator = torch.Generator().manual_seed(185)
    weight = (torch.randn(4096, 24896, generator=generator) * 0.02).bfloat16().cuda()
    x = (torch.randn(rows, 4096, generator=generator) * 0.25).bfloat16().cuda()
    module = _model_module(weight)
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    actual = projection.quant_method.apply(projection, x)
    # Compute the oracle with FP32 accumulation, then round once to BF16.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    reference = (x.float() @ weight.float()).bfloat16()
    error = actual.float() - reference.float()
    assert actual.shape == (rows, 24896)
    assert error.square().mean().sqrt().item() < 2e-4


@pytest.mark.parametrize("rows", [0, 1, 15, 16, 17, 31, 32])
@pytest.mark.parametrize("is_mtp", [False, True])
def test_glm5next_projection_preserves_rows_and_bias(optional_ops, rows, is_mtp):
    weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    module = _model_module(weight)
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0, is_mtp_layer=is_mtp).proj
    x = torch.tensor([[1, 2]], dtype=torch.bfloat16).repeat(rows, 1)
    bias = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    actual = projection.quant_method.apply(projection, x, bias)
    expected = torch.tensor([[10, 14, 18]], dtype=torch.bfloat16).repeat(rows, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("disabled", ["master", "gfx938", "nn"])
def test_glm5next_bf16_binding_obeys_each_guard(optional_ops, monkeypatch, disabled):
    if disabled == "master":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    elif disabled == "gfx938":
        monkeypatch.setattr(hcu, "on_gfx938", lambda: False)
    else:
        monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16))
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" not in vars(projection.quant_method)


def test_glm5next_binding_is_instance_local_and_idempotent(optional_ops):
    from vllm_hcu.model_executor.layers.linear import UnquantizedLinearMethod

    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16))
    unrelated = UnquantizedLinearMethod()
    patch.apply_to_module(module)
    constructor = module.Glm5NextDecoderLayer.__init__
    assert not patch.apply_to_module(module)
    assert module.Glm5NextDecoderLayer.__init__ is constructor
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" in vars(projection.quant_method)
    assert "apply" not in vars(unrelated)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_glm5next_other_dtypes_keep_original_method(optional_ops, dtype):
    module = _model_module(torch.ones(2, 3, dtype=dtype))
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" not in vars(projection.quant_method)


def test_glm5next_non_matrix_input_preserves_broadcast(optional_ops):
    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16))
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    x = torch.ones(2, 5, 2, dtype=torch.bfloat16)
    actual = projection.quant_method.apply(projection, x)
    torch.testing.assert_close(actual, torch.full((2, 5, 3), 2, dtype=x.dtype))


def test_glm5next_row_dependent_bias_preserves_original_shape(optional_ops):
    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16))
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    x = torch.ones(5, 2, dtype=torch.bfloat16)
    bias = torch.arange(15, dtype=x.dtype).reshape(5, 3)
    actual = projection.quant_method.apply(projection, x, bias)
    torch.testing.assert_close(actual, bias + 2, rtol=0, atol=0)


def test_glm5next_transposed_weight_keeps_original_method(optional_ops):
    module = _model_module(torch.ones(3, 2, dtype=torch.bfloat16).t())
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" not in vars(projection.quant_method)


def test_glm5next_quantized_method_keeps_original_apply(optional_ops):
    class QuantizedMethod:
        def apply(self, layer, x, bias=None):
            return x @ layer.weight

    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16), QuantizedMethod)
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" not in vars(projection.quant_method)


def test_glm5next_batch_invariant_mode_keeps_original_method(optional_ops, monkeypatch):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    module = _model_module(torch.ones(2, 3, dtype=torch.bfloat16))
    patch.apply_to_module(module)
    projection = module.Glm5NextDecoderLayer(None, None, 0).proj
    assert "apply" not in vars(projection.quant_method)
