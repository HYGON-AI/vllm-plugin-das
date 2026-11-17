# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only seam tests; these do not claim vLLM imports or HCU execution."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FLASH = ROOT / "vllm_hcu/models/hy_v4_flash"


def runtime(monkeypatch, hcu=True):
    cls = type("Platform", (), {"__module__": "vllm_hcu.platforms.hcu" if hcu else "vllm.platforms.cuda"})
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = cls()
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    spec = importlib.util.spec_from_file_location("_hy4_flash_runtime_contract", FLASH / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hcu_provider_does_not_require_deepgemm(monkeypatch):
    module = runtime(monkeypatch)
    pair = (object(), object())
    bridge = ModuleType("hcu_adapter.bridges.vllm_paged_mqa")
    bridge.get_paged_mqa_provider = lambda: pair
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    assert module.get_paged_mqa_provider() == pair


def test_hcu_missing_adapter_reports_dependency(monkeypatch):
    module = runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "hcu_adapter.bridges.vllm_paged_mqa", None)
    with pytest.raises(RuntimeError, match="requires adapter-das"):
        module.get_paged_mqa_provider()


@pytest.mark.parametrize("available", [True, False])
def test_cuda_provider_still_uses_vllm(monkeypatch, available):
    module = runtime(monkeypatch, hcu=False)
    deepgemm = ModuleType("vllm.utils.deep_gemm")
    deepgemm.has_deep_gemm = lambda: available
    deepgemm.get_paged_mqa_logits_metadata = object()
    deepgemm.fp8_fp4_paged_mqa_logits = object()
    monkeypatch.setitem(sys.modules, deepgemm.__name__, deepgemm)
    if available:
        assert module.get_paged_mqa_provider() == (deepgemm.get_paged_mqa_logits_metadata, deepgemm.fp8_fp4_paged_mqa_logits)
    else:
        with pytest.raises(RuntimeError, match="requires DeepGEMM"):
            module.get_paged_mqa_provider()


def test_v033_mx_inputs_call_has_three_arguments():
    tree = ast.parse((FLASH / "nvidia/compressed_indexer.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "mqa_logits_mx_inputs"]
    assert len(calls) == 1
    call = compile(ast.Expression(calls[0]), "<v033-mx-call>", "eval")
    def v033_inputs(q, q_scale, weights):
        return q_scale, weights
    assert eval(call, {"mqa_logits_mx_inputs": v033_inputs, "quant_q": object(), "weights": 3}) == (None, 3)
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Attribute) and target.attr == "use_mx_scales" for target in node.targets))
    def forbidden():
        raise AssertionError("HCU must not infer MX scales from CUDA capability")
    assert eval(compile(ast.Expression(assignment.value), "<hcu-non-mx>", "eval"),
                {"is_hcu_platform": lambda: True, "mqa_logits_use_mx_scales": forbidden}) is False


@pytest.mark.parametrize("skip_topk", [False, True])
def test_flashmla_orchestration_needs_no_rocm_event(skip_topk):
    tree = ast.parse((FLASH / "nvidia/attention.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_indexer_and_attn")
    assert [item.id for item in method.decorator_list] == ["eager_break_during_capture"]
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<flash-attn-seam>", "exec"), namespace)
    calls = []
    result = object()
    class Output:
        shape = (1, 2)
        def copy_(self, value):
            assert value is result
            calls.append("copy")
    def indexer(*args):
        calls.append("indexer")
    def attention(*args, **kwargs):
        assert kwargs == {"output_shape": (1, 2)}
        calls.append("attention")
        return result
    model = SimpleNamespace(indexer=indexer, is_sparse=True, skip_topk=skip_topk,
                            indexer_rope_emb=None, mla_attn=attention)
    namespace["_indexer_and_attn"](model, None, None, None, None, None, None, Output())
    assert calls == (["indexer"] if not skip_topk else []) + ["attention", "copy"]
