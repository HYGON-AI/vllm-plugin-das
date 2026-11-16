# SPDX-License-Identifier: Apache-2.0
"""Exercise Step5 MoE admission with the real process master predicate."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
MOE = "vllm_hcu/model_executor/layers/fused_moe/"
_env_spec = importlib.util.spec_from_file_location(
    "step5_moe_master_test_envs", ROOT / "vllm_hcu/platforms/envs.py")
HENVS = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(HENVS)


def load_function(relative, name, **namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    scope = dict(os=os, henvs=HENVS, **namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, fn], type_ignores=[])), relative, "exec"), scope)
    return scope[name]


def load_assignments(relative, names):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes) == len(names)
    scope = dict(os=os, henvs=HENVS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), scope)
    return {name: scope[name] for name in names}


@pytest.mark.parametrize("master", ["0", "false", "False", "1", "true"])
def test_step5_gemm_and_communication_options_follow_master(monkeypatch, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    for flag in ("HCU_STEP5_EXPERT_GRID", "HCU_STEP5_PREFILL_G2",
                 "HCU_STEP5_PREFILL_ASM_G1", "HCU_DP_MOE_PIPE_UNIFORM"):
        monkeypatch.setenv(flag, "1")
    for flag in ("HCU_EP_DIRECT_SUM", "HCU_DP_MOE_PIPE_MIN_TOKENS",
                 "HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS"):
        monkeypatch.setenv(flag, "8192")
    enabled = master in ("1", "true")
    ep = load_assignments(MOE + "ep_fused_moe.py", {
        "STEP5_EXPERT_GRID", "STEP5_PREFILL_G2", "STEP5_PREFILL_ASM_G1", "DIRECT_MIN"})
    assert ep == dict(STEP5_EXPERT_GRID=enabled, STEP5_PREFILL_G2=enabled,
                     STEP5_PREFILL_ASM_G1=enabled, DIRECT_MIN=8192 if enabled else 0)
    pipe = load_assignments(MOE + "modular_kernel.py", {
        "_DP_MOE_PIPE_MIN_TOKENS", "_EP_DIRECT_SUM", "_DP_PIPE_UNIFORM"})
    assert pipe == dict(_DP_MOE_PIPE_MIN_TOKENS=8192 if enabled else 0,
                       _EP_DIRECT_SUM=8192 if enabled else 0, _DP_PIPE_UNIFORM=enabled)
    shared = load_assignments(MOE + "shared_experts.py", {"_LATE_STREAM_MIN_TOKENS"})
    assert shared["_LATE_STREAM_MIN_TOKENS"] == (8192 if enabled else 0)


@pytest.mark.parametrize("relative,names,flags,defaults", [
    ("ep_fused_moe.py", {"DIRECT_MIN", "EP_SORT_MAX_ROUTES", "EP_TABLE_PATH", "CHUNK"},
     ("HCU_EP_DIRECT_SUM", "EP_FUSED_MOE_SORT_MAX_ROUTES", "EP_FUSED_MOE_TABLE",
      "EP_FUSED_MOE_CHUNK"),
     dict(DIRECT_MIN=0, EP_SORT_MAX_ROUTES=2048, EP_TABLE_PATH="", CHUNK=32768)),
    ("modular_kernel.py", {"_DP_MOE_PIPE_MIN_TOKENS", "_EP_DIRECT_SUM"},
     ("HCU_DP_MOE_PIPE_MIN_TOKENS", "HCU_EP_DIRECT_SUM"),
     dict(_DP_MOE_PIPE_MIN_TOKENS=0, _EP_DIRECT_SUM=0)),
    ("shared_experts.py", {"_LATE_STREAM_MIN_TOKENS"},
     ("HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS",), dict(_LATE_STREAM_MIN_TOKENS=0)),
])
def test_disabled_master_does_not_parse_optional_thresholds(
    monkeypatch, relative, names, flags, defaults,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    for flag in flags:
        monkeypatch.setenv(flag, "invalid-unused-configuration")
    assert load_assignments(MOE + relative, names) == defaults


@pytest.mark.parametrize("master", ["0", "1"])
@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("compact", ["0", "1"])
def test_compact_rows_requires_master_and_step5_owner(monkeypatch, master, owner, compact):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("HCU_STEP5_DEEPEP_COMPACT_ROWS", compact)
    relative = MOE + "experts/deep_gemm_moe.py"
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "DeepGemmExperts")
    ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    nodes = [n for n in ctor.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr in (
            "_hcu_step5_owner", "_hcu_step5_compact_ht") for t in n.targets)]
    expert = SimpleNamespace()
    scope = dict(self=expert, moe_config=object(), is_step5_owner=lambda _: owner,
                 os=os, henvs=HENVS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), scope)
    assert expert._hcu_step5_owner is owner
    assert expert._hcu_step5_compact_ht is (master == "1" and owner and compact == "1")


def test_disabled_master_never_enters_optional_deepep_construction(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    for flag in ("HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT", "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR",
                 "HCU_STEP5_DEEPEP_FUSED_SCATTER"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setenv("HCU_DBO_MIN_TOKENS", "invalid-unused-threshold")
    configure = load_function(MOE + "deepep_runtime.py", "configure_step5_ht_kernel")
    kernel = SimpleNamespace()
    assert configure(kernel, None, None) is kernel
    assert vars(kernel) == {}


def test_disabled_master_ignores_runtime_profile_even_with_cached_owner(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("HCU_STEP5_DEEPEP_RUNTIME_CONFIG", "/must-not-be-read.json")
    apply = load_function(MOE + "deepep_runtime.py", "_apply_step5_ht_runtime_config")
    launch = load_function(MOE + "deepep_runtime.py", "step5_ht_launch_config")
    prepare = SimpleNamespace(_hcu_step5_runtime_tuning_allowed=True)
    apply(prepare)
    original = object()
    assert launch(prepare, original) is original
    assert vars(prepare) == {"_hcu_step5_runtime_tuning_allowed": True}


def test_disabled_master_cannot_use_cached_pipe_or_late_stream_flags(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    pipe = load_function(MOE + "modular_kernel.py", "_dp_pipe_apply")
    assert pipe(*[None] * 10) is None
    late = load_function(MOE + "shared_experts.py", "_late_stream",
                         _LATE_STREAM_MIN_TOKENS=8192, dbo_enabled=lambda: False)
    expert = SimpleNamespace(_hcu_step5_owner=True, _stream=object())
    assert late(expert, SimpleNamespace(shape=(65536, 4096))) is False


def test_disabled_master_never_selects_private_workspace_provider(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("HCU_STEP5_BOUNDED_EP_WORKSPACE", "1")
    provider = load_function(MOE + "modular_kernel.py", "_step5_ep_chunk_provider")
    assert provider(None, None, None, None) is None


def test_disabled_master_does_not_consult_cached_external_ep_recipe(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    def forbidden(*args):
        raise AssertionError("disabled master read the external EP recipe")
    pick = load_function(MOE + "ep_fused_moe.py", "pick_configs",
                         _device_arch=lambda _: "gfx938", _ep_table=forbidden,
                         lib_configs=lambda *args: "library-default")
    weight = SimpleNamespace(shape=(22, 3072, 4096), device=SimpleNamespace(index=0))
    assert pick(weight, object(), 8, 65536) == "library-default"
