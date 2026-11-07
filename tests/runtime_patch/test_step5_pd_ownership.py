from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ownership():
    path = ROOT / "vllm_hcu/runtime_compat/step5.py"
    spec = importlib.util.spec_from_file_location("step5_pd_ownership_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["step5", "step3p5", "deepseek_v3", "qwen3", "llama", "unknown"])
@pytest.mark.parametrize("connector", ["MooncakeConnector", "Step5MooncakeConnector", "NixlConnector", None])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both", None])
def test_only_step5_separate_mooncake_roles_opt_in(ownership, model, connector, role):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
        kv_transfer_config=SimpleNamespace(kv_connector=connector, kv_role=role),
    )
    expected = model == "step5" and connector in ("MooncakeConnector", "Step5MooncakeConnector") and role in ("kv_producer", "kv_consumer")
    assert ownership.is_step5_pd_config(config) is expected


def test_absent_config_and_ifb_do_not_opt_in(ownership):
    assert not ownership.is_step5_pd_config(None)
    assert not ownership.is_step5_pd_config(SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="step5"))))


def test_nested_step5_text_model_opts_in(ownership):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            model_type="step5_vl", text_config=SimpleNamespace(model_type="step5"))),
        kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector", kv_role="kv_producer"),
    )
    assert ownership.is_step5_pd_config(config)


def test_shared_dsa_preserves_original_aliases_without_pd_opt_in():
    path = ROOT / "vllm_hcu/model_executor/layers/step4_dsa.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Step4DSAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_write_kv")
    method.decorator_list = []
    node = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    calls = []
    namespace = {"torch": torch, "_store_kv_rows": lambda *args: calls.append(args)}
    exec(compile(ast.fix_missing_locations(node), str(path), "exec"), namespace)
    cache = torch.zeros((3, 4, 8, 8), dtype=torch.bfloat16)
    empty = torch.empty((0, 4, 4));slots = torch.empty(0, dtype=torch.long)
    k, v = namespace['_write_kv'](cache, empty, empty, slots)
    assert k.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert k.stride() == (cache.stride(2), cache.stride(1), 1)
    assert len(calls) == 1
