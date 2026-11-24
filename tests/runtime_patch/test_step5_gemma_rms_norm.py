"""Step5-only library dispatch; other owners retain the existing adapter."""
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.ops import gemma_rms_norm as norms


def owner(step5=True):
    return SimpleNamespace(_hcu_step5_owner=step5,
        weight=SimpleNamespace(dtype=torch.float32), variance_epsilon=1e-6,
        forward_native=lambda *args: "native")


def tensor(width=4096, cuda=True, contiguous=True):
    return SimpleNamespace(shape=(64, width), dtype=torch.bfloat16, is_cuda=cuda,
                           is_contiguous=lambda: contiguous, clone=lambda: "original-output")


@pytest.mark.parametrize("width", [192, 4096])
def test_step5_routes_mixed_dtype_without_casting_gamma(monkeypatch, width):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    layer, x, calls = owner(), tensor(width), []
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: calls.append(args) or "library")
    assert norms.HcuGemmaRMSNorm.forward_hip(layer, x) == "library"
    assert calls == [(x, layer.weight, layer.variance_epsilon)]
    assert layer.weight.dtype == torch.float32


@pytest.mark.parametrize("kwargs", [dict(width=256), dict(cuda=False), dict(contiguous=False)])
def test_step5_unsupported_layout_keeps_native_semantics(monkeypatch, kwargs):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: pytest.fail("unsupported library invocation"))
    assert norms.HcuGemmaRMSNorm.forward_hip(owner(), tensor(**kwargs)) == "native"


def test_master_off_never_probes_input_or_library(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    assert norms.HcuGemmaRMSNorm.forward_hip(owner(), object()) == "native"


def test_other_models_keep_original_lightop_adapter(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    calls = []
    monkeypatch.setattr(torch.ops.vllm, "hcu_gemma_rms_norm",
                        lambda *args: calls.append(args) or "original-output")
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: pytest.fail("Step5 dispatch leaked"))
    layer, x = owner(False), tensor()
    assert norms.HcuGemmaRMSNorm.forward_hip(layer, x) == "original-output"
    assert calls == [(x, layer.weight, layer.variance_epsilon)]


@pytest.mark.parametrize("step5,selected,master", [
    (True, True, True), (True, False, True), (False, True, True), (True, True, False)])
@pytest.mark.parametrize("decode_profile", [False, True])
def test_model_constructor_preserves_fp32_and_explicit_provider(monkeypatch, step5, selected, master, decode_profile):
    from vllm_hcu.runtime_compat import step5 as step5_context
    monkeypatch.setattr(step5_context, 'is_step5_tp4_deepep_ll', lambda: decode_profile)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1" if master else "0")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")

    class Norm:
        def __init__(self, *args):
            self._hcu_step5_owner = step5
            self._forward_method = self.forward_hip if selected else self.forward_native

        def forward_hip(self, *args):
            pass

        def forward_native(self, *args):
            pass

    path = Path(__file__).parents[2] / 'vllm_hcu/models/step5_text.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_fp32_gemma_rms_norm')
    namespace = dict(GemmaRMSNorm=Norm, torch=torch, nn=torch.nn, henvs=norms.henvs)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        layer = namespace['_fp32_gemma_rms_norm'](4096, 1e-6)
    finally:
        torch.set_default_dtype(previous)
    assert layer.weight.dtype == torch.float32
    expected = 'forward_hip' if decode_profile and step5 and selected and master else 'forward_native'
    assert layer._forward_method.__name__ == expected
