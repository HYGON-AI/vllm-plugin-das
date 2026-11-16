# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Step5 worker child switches under the custom-op master."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _nodes(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _compile(nodes, namespace):
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<Step5 runtime>", "exec"), namespace)
    return namespace


def _envs():
    names = {"_environment_flag", "custom_ops_enabled", "optional_custom_op_enabled"}
    nodes = [node for node in _nodes("vllm_hcu/platforms/envs.py").body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    return SimpleNamespace(**_compile(nodes, {"os": os}))


@pytest.mark.parametrize("master", ["0", "false", "1", "true"])
@pytest.mark.parametrize("path,constant,child", [
    ("vllm_hcu/v1/hcu_dbo.py", "DBO_MIN_TOKENS", "HCU_DBO_MIN_TOKENS"),
    ("vllm_hcu/v1/step5_hot_context.py", "HOT_REGIONS", "HCU_STEP5_CONTEXT_GRAPH_REGIONS"),
    ("vllm_hcu/patch/worker/framework_opt/patch_mtp_prefill_tail.py", "_TAIL", "HCU_MTP_PREFILL_TAIL"),
])
def test_worker_numeric_children_are_inert_under_disabled_master(
    monkeypatch, master, path, constant, child
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    enabled = master in ("1", "true")
    # An inactive optimization must not parse a stale launch-script value.
    monkeypatch.setenv(child, "8192" if enabled else "unused")
    nodes = [node for node in _nodes(path).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == constant for t in node.targets)]
    actual = _compile(nodes, {"os": os, "henvs": _envs()})[constant]
    assert actual == (8192 if enabled else 0)


def test_mtp_tail_cached_opt_in_cannot_bypass_master(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    fn = next(node for node in _nodes(
        "vllm_hcu/patch/worker/framework_opt/patch_mtp_prefill_tail.py"
    ).body if isinstance(node, ast.FunctionDef) and node.name == "_tail_enabled")
    runtime = _compile([fn], {"henvs": _envs()})
    assert runtime[fn.name](SimpleNamespace(_hcu_mtp_tail_ok=True)) is False


@pytest.mark.parametrize("master", ["0", "1"])
def test_compact_host_tables_obey_master(monkeypatch, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("HCU_STEP5_COMPACT_HOST_BLOCK_TABLES", "1")
    tree = _nodes("vllm_hcu/patch/worker/framework_opt/patch_step5_block_tables.py")
    names = {"enable_host_mirror", "gather_block_tables_np"}
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    runtime = _compile(functions, {"np": np, "os": os, "henvs": _envs()})
    tables = SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.zeros((1, 2), dtype=np.int32)),
        block_tables=[SimpleNamespace(gpu=SimpleNamespace(shape=(2, 8)))],
    )
    runtime["enable_host_mirror"](tables)
    tables.num_blocks.np[0, 0] = 2
    actual = runtime["gather_block_tables_np"](tables, np.array([0]), 1)[0]
    assert actual.shape == (1, 2 if master == "1" else 8)


@pytest.mark.parametrize("master,model,expected", [
    ("0", "step5", "native"),
    ("1", "step5", "step5"),
    ("1", "qwen3", "native"),
])
def test_runner_master_off_delegates_to_native_execute(monkeypatch, master, model, expected):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    calls = []

    class NativeRunner:
        def execute_model(self, *args, **kwargs):
            calls.append(("native", args, kwargs))
            return "native"

    optimized = ModuleType("vllm_hcu.v1.step5_execution")

    def optimized_execute(self, *args, **kwargs):
        calls.append(("step5", args, kwargs))
        return "step5"

    optimized.execute_model = optimized_execute
    monkeypatch.setitem(sys.modules, optimized.__name__, optimized)
    tree = _nodes("vllm_hcu/v1/hcu_model_runner_v2.py")
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "HcuGPUModelRunnerV2")
    execute = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute_model")
    execute.decorator_list = []
    subject = ast.ClassDef(
        name=cls.name, bases=[ast.Name("NativeRunner", ast.Load())],
        keywords=[], body=[execute], decorator_list=[],
    )
    runtime = _compile([subject], {
        "NativeRunner": NativeRunner,
        "nullcontext": nullcontext,
        "deepep_auto_request_phase_scope": nullcontext,
        "_uses_global_pcp_batch": lambda config: False,
        "_DEEPEP_AUTO_DUMMY_PHASE": SimpleNamespace(get=lambda: None),
        "henvs": _envs(),
        "is_step5_config": lambda config: config.model == "step5",
    })
    runner = runtime[cls.name]()
    runner.vllm_config = SimpleNamespace(model=model)
    assert runner.execute_model("scheduler", dummy_run=False) == expected
    assert calls == [(expected, ("scheduler",), {"dummy_run": False})]


@pytest.mark.parametrize("master", ["0", "1"])
def test_fp8_oracle_dbo_child_cannot_configure_with_master_off(monkeypatch, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT", "1")
    tree = _nodes("vllm_hcu/patch/worker/op_opt/moe/patch_fp8_oracle.py")
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "hcu_make_fp8_moe_kernel")
    fn.decorator_list = []
    configured = []
    runtime_module = ModuleType("vllm_hcu.model_executor.layers.fused_moe.deepep_runtime")
    runtime_module.configure_step5_ht_kernel = lambda *args: configured.append(args)
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    kernel = object()
    runtime = _compile([fn], {
        "os": os, "henvs": _envs(),
        "hcu_enum": SimpleNamespace(HCU_DEEPGEMM=object()),
        "make_kernel": lambda *args: kernel,
    })
    quant = SimpleNamespace()
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_deepep_auto_kernels=False))
    assert runtime[fn.name](quant, config, object(), object()) is kernel
    assert configured == ([(kernel, quant, config)] if master == "1" else [])
