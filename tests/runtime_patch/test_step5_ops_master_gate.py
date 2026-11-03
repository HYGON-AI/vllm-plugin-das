# SPDX-License-Identifier: Apache-2.0
"""CPU policy checks for Step5 tuning and kernel-entry master gates."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_hcu/ops/step5"


@pytest.fixture
def henvs():
    # Load the real dependency-light policy without importing GPU packages.
    path = ROOT / "vllm_hcu/platforms/envs.py"
    spec = importlib.util.spec_from_file_location("step5_ops_test_envs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fragment(name, henvs, *, symbols=None, **namespace):
    path = OPS / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if symbols is None:
        nodes = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.module == "vllm_hcu.platforms"
        )]
    else:
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in symbols:
                node.decorator_list = []
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in symbols for t in node.targets
            ):
                nodes.append(node)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future] + nodes, type_ignores=[])
    namespace.update(henvs=henvs, os=os)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("master", ["0", "false", "False"])
def test_disabled_master_ignores_external_tuning_path(monkeypatch, henvs, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", "/missing/step5-recipe.json")
    tuning = fragment("tuning.py", henvs)
    assert tuning.tuning_data() == {}
    assert tuning.dense_recipe() is None
    assert tuning.asm_recipe(22) is None


def test_cached_recipes_cannot_bypass_disabled_master(monkeypatch, tmp_path, henvs):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "architecture": "gfx938",
        "dense": {"hipblaslt_sha256": "a" * 64, "algorithms": {"8,16,32": 7}},
        "asm_g1": {
            "block_m": 128, "solution": 7,
            "persist_by_local_experts": {"22": 16, "23": 32},
        },
    }))
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", str(path))
    tuning = fragment("tuning.py", henvs)
    expected_dense = ({(8, 16, 32): 7}, "a" * 64)
    assert tuning.dense_recipe() == expected_dense
    assert tuning.asm_recipe(22) == (7, 16)
    assert tuning.asm_recipe(23) == (7, 32)

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", "/missing/step5-recipe.json")
    assert tuning.tuning_data() == {}
    assert tuning.dense_recipe() is None
    assert tuning.asm_recipe(22) is None
    assert tuning.asm_recipe(23) is None

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", str(path))
    assert tuning.dense_recipe() == expected_dense
    assert tuning.asm_recipe(22) == (7, 16)


def test_enabled_master_keeps_tuning_validation(monkeypatch, tmp_path, henvs):
    path = tmp_path / "recipe.json"
    path.write_text('{"schema_version": 0}')
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG", str(path))
    tuning = fragment("tuning.py", henvs)
    with pytest.raises(ValueError, match="schema_version"):
        tuning.tuning_data()


@pytest.mark.parametrize("master,child,expected", [
    ("0", "1", False), ("false", "1", False),
    ("1", "1", True), ("true", "1", True), ("1", "0", False),
])
def test_dense_and_asm_child_flags_obey_master(
    monkeypatch, henvs, master, child, expected
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("HCU_STEP5_DENSE_TUNING", child)
    monkeypatch.setenv("HCU_STEP5_PREFILL_ASM_G1", child)
    dense = fragment("fp8_dense.py", henvs, symbols={"_TUNED_DENSE"})
    asm = fragment("prefill_asm_g1.py", henvs, symbols={"_enabled"})
    assert dense._TUNED_DENSE is expected
    assert asm._enabled() is expected


@pytest.mark.parametrize("value", ["not-an-int", "-1", "4096"])
def test_router_threshold_is_not_parsed_when_master_disabled(
    monkeypatch, henvs, value
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("HCU_STEP5_ROUTER_SHARED_MAX_M", value)
    router = fragment("router_gemm.py", henvs, symbols={"RSG_MAX_M"})
    assert router.RSG_MAX_M == 0


@pytest.mark.parametrize("value", ["0", "16", "32", "64"])
def test_router_threshold_is_preserved_when_master_enabled(
    monkeypatch, henvs, value
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("HCU_STEP5_ROUTER_SHARED_MAX_M", value)
    router = fragment("router_gemm.py", henvs, symbols={"RSG_MAX_M"})
    assert router.RSG_MAX_M == int(value)


def test_disabled_asm_returns_before_inspecting_weights(monkeypatch, henvs):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("HCU_STEP5_PREFILL_ASM_G1", "1")
    asm = fragment("prefill_asm_g1.py", henvs, symbols={
        "_enabled", "prepare_prefill_asm_g1_cache", "try_step5_prefill_asm_g1",
    })
    assert asm.prepare_prefill_asm_g1_cache(object(), object(), object()) is False
    assert asm.try_step5_prefill_asm_g1(
        *(object() for _ in range(10)), num_tokens=65536, top_k=8,
        sort_block_m=256, limit=7.0, workspace2=object(),
    ) is None


def test_cached_dense_entry_returns_before_touching_inputs(monkeypatch, henvs):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    tuned = fragment("fp8_dense_tuned.py", henvs, symbols={"tuned_fp8_gemm"})
    assert tuned.tuned_fp8_gemm(object(), object(), object(), object()) is None


def test_dense_child_cached_on_uses_fallback_after_master_is_disabled(
    monkeypatch, henvs
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    out = SimpleNamespace(view=lambda *shape: "fallback")
    calls = []
    lightop = ModuleType("lightop.gemm_ops")

    def gemm(*args):
        calls.append("fallback")
        return True, out

    lightop.hipblaslt_w8a8_channelwise_gemm = gemm
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", lightop)
    dense = fragment(
        "fp8_dense.py", henvs, symbols={"_step5_fp8_gemm"},
        _TUNED_DENSE=True, torch=SimpleNamespace(bfloat16=object()),
    )
    x = SimpleNamespace(shape=(2, 16))
    weight = SimpleNamespace(shape=(32, 16))
    assert dense._step5_fp8_gemm(x, object(), weight, object()) == "fallback"
    assert calls == ["fallback"]
