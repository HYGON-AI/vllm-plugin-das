# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contract tests for the Step5 EP activation/quantization ownership boundary.

The EP gated-SWIGLUSTEP + FP8 quantization kernel lives in BoltOps
(``boltops.fused_moe.triton.moe_activation.ep_swiglustep_quant``). These tests
pin the boundary from the plugin side:

* the plugin does not carry a second copy of the kernel and imports the op;
* the EP pipeline still routes through the module-level ``ep_swiglustep_quant``
  global, which is what lets the existing CPU contract tests inject a stub;
* the BoltOps wrapper keeps the launch contract the plugin relies on
  (caller-owned ``q``, grid bounds, kernel arguments);
* the kernel keeps the semantics the EP path relies on, proven on CPU through
  Triton's NumPy interpreter (``ep_activation_cpu_probe.py``).

The numeric identity of the move itself is a review-time check: the pre-patch
kernel from git history and the BoltOps kernel are executed on the same inputs
under Triton's NumPy interpreter and agree bit for bit in ``q`` and ``scale``
(see the merge request description for the command and the case matrix).
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

MODULE = "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py"
ROOT = pathlib.Path(__file__).parents[2]
FP8 = torch.float8_e4m3fn


def module_tree():
    return ast.parse((ROOT / MODULE).read_text())


def functions(tree):
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def load_boltops_activation():
    """Load BoltOps' activation module by path; its package __init__ queries devices."""
    import boltops

    path = pathlib.Path(boltops.__file__).parent / "fused_moe/triton/moe_activation.py"
    spec = importlib.util.spec_from_file_location("_step5_ep_activation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_does_not_duplicate_the_ep_activation_kernel():
    defined = functions(module_tree())
    assert "ep_swiglustep_quant" not in defined
    assert "_ep_swiglustep_quant_kernel" not in defined
    # Still required by the DeepGEMM Step5 variant and by prefill_asm_g1.
    assert "_ep_swiglu_chunk" in defined
    assert "_ep_swiglustep_quant_deepgemm_kernel" in defined


def test_plugin_imports_the_op_from_boltops():
    tree = module_tree()
    imported = [
        (node.module, {alias.name for alias in node.names})
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert ("boltops.fused_moe.triton", {"moe_activation"}) in imported
    bound = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ep_swiglustep_quant"
                for target in node.targets)
    ]
    assert len(bound) == 1, "the op must be bound from the BoltOps module"
    value = bound[0].value
    assert isinstance(value, ast.Attribute) and value.attr == "ep_swiglustep_quant"
    assert isinstance(value.value, ast.Name) and value.value.id == "_boltops_moe_activation"


def test_ep_pipeline_calls_the_boltops_op_with_a_workspace_buffer():
    pipeline = functions(module_tree())["_ep_fused_experts"]
    calls = [
        node for node in ast.walk(pipeline)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ep_swiglustep_quant"
    ]
    assert len(calls) == 1, "the EP pipeline must call the op exactly once"
    call = calls[0]
    assert len(call.args) == 5
    assert [getattr(arg, "id", None) for arg in call.args[:4]] == [
        "cache1", "sorted_w", "nvalid", "rows"]
    assert isinstance(call.args[4], ast.IfExp), "gemm1_limit is the activation clamp"
    assert [keyword.arg for keyword in call.keywords] == ["q"], (
        "act_q must come from the caller-owned workspace")


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize("rows,programs,expected_grid", [
    (100, 4, (50,)),     # cdiv(rows, 2) below the CTA cap
    (1000, 4, (256,)),   # CTA cap: 4 CUs * 64 CTAs
    (0, 4, (1,)),        # never launch an empty grid
])
def test_boltops_wrapper_launch_contract(monkeypatch, rows, programs, expected_grid):
    activation = load_boltops_activation()
    recorder = _Recorder()
    monkeypatch.setattr(activation, "_ep_swiglustep_quant_kernel", recorder)
    monkeypatch.setattr(activation, "_ep_num_cus", lambda device: programs)

    cache1 = torch.empty((max(rows, 1), 2 * 1536), dtype=torch.bfloat16)
    weights = torch.empty(max(rows, 1), dtype=torch.float32)
    nvalid = torch.empty(1, dtype=torch.int32)
    caller_q = torch.empty((max(rows, 1), 1536), dtype=FP8)

    q, scale = activation.ep_swiglustep_quant(
        cache1, weights, nvalid, rows, limit=7.0, q=caller_q)

    assert q is caller_q, "a caller-owned q must be used in place"
    assert scale.shape == (rows, 1) and scale.dtype == torch.float32
    grid, args, kwargs = recorder.calls[0]
    assert grid == expected_grid
    assert args[0] is caller_q and args[1] is scale
    assert args[2] is cache1 and args[3] is weights and args[4] is nvalid
    assert kwargs == {
        "D": 1536, "LIMIT": 7.0, "DTYPE_MAX": float(torch.finfo(FP8).max),
        "BLOCK_M": 2, "CH": 512, "num_warps": 1, "num_stages": 1,
    }


def test_boltops_wrapper_allocates_q_when_the_caller_does_not(monkeypatch):
    activation = load_boltops_activation()
    monkeypatch.setattr(activation, "_ep_swiglustep_quant_kernel", _Recorder())
    monkeypatch.setattr(activation, "_ep_num_cus", lambda device: 4)
    cache1 = torch.empty((7, 2 * 1536), dtype=torch.bfloat16)
    weights = torch.empty(7, dtype=torch.float32)
    nvalid = torch.empty(1, dtype=torch.int32)

    q, scale = activation.ep_swiglustep_quant(cache1, weights, nvalid, 7)

    assert q.shape == (7, 1536) and q.dtype == FP8 and q.is_contiguous()
    assert scale.shape == (7, 1) and scale.dtype == torch.float32


def test_boltops_wrapper_rejects_widths_that_cannot_be_split_in_three(monkeypatch):
    activation = load_boltops_activation()
    monkeypatch.setattr(activation, "_ep_num_cus", lambda device: 4)
    cache1 = torch.empty((4, 2 * 1024), dtype=torch.bfloat16)  # D = 1024, CH = 341.33
    weights = torch.empty(4, dtype=torch.float32)
    nvalid = torch.empty(1, dtype=torch.int32)
    with pytest.raises(AssertionError):
        activation.ep_swiglustep_quant(cache1, weights, nvalid, 4)


def test_cpu_kernel_semantics():
    """Interpreter-mode proof of the runtime contract; runs in a child process."""
    probe = pathlib.Path(__file__).with_name("ep_activation_cpu_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ALL PASS" in result.stdout
