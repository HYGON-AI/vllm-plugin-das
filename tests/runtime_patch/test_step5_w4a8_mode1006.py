# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only guard rails for the opt-in Step5 W4A8 MODE1006 prefill path.

The runtime code is unchanged by these tests; they parse the two source
files and assert the reviewed invariants without importing torch/triton.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOE = ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_w4a8.py"
ROUTING = (
    ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_w4a8_routing.py"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def test_experiment_is_opt_in_and_prefill_only_named():
    tree = _tree(MOE)
    allowed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id != "experiment":
                continue
            for cmp in node.comparators:
                if isinstance(cmp, (ast.Tuple, ast.List)):
                    values = [
                        e.value for e in cmp.elts
                        if isinstance(e, ast.Constant)
                    ]
                    if values:
                        allowed = values
    assert allowed == ["native", "agrs_deepgemm1006"]


def test_mode1006_chunk_stays_below_32bit_descriptor():
    tree = _tree(MOE)
    value = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_MODE1006_MAX_ROWS"
                for t in node.targets
            )
        ):
            value = eval(
                ast.unparse(node.value), {"__builtins__": {}}, {}
            )
    assert value == (1 << 32) // 4096 - 128
    assert value % 128 == 0
    assert value * 4096 < 1 << 32


def test_both_gemm_call_sites_use_the_large128_gate():
    tree = _tree(MOE)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "native_gemm"
    ]
    passed = [
        kw for call in calls for kw in call.keywords
        if kw.arg == "use_deepgemm_gemm"
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "use_large128"
    ]
    assert len(passed) == 2


def test_permute_aligned_padding_and_default_path():
    tree = _tree(ROUTING)
    funcs = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "permute_aligned" in funcs
    assert "permute64" in funcs
    src = ast.get_source_segment(
        ROUTING.read_text(), funcs["permute64"]
    )
    assert "permute_aligned" in src and "alignment=64" in src
    src = ROUTING.read_text()
    assert "indices = torch.full(" in src
    assert "-1" in src
