# SPDX-License-Identifier: Apache-2.0
"""CPU policy regressions for Step5 placement and DSA performance options."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "vllm_hcu/models/step5_text.py"
DSA = ROOT / "vllm_hcu/model_executor/layers/step4_dsa.py"


def _policy(path, names, *, environ=None):
    """Execute the real startup policy without importing GPU providers."""
    helpers = ast.parse(
        (ROOT / "vllm_hcu/platforms/envs.py").read_text(encoding="utf-8")
    )
    selected = [
        n for n in helpers.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_environment_flag", "custom_ops_enabled"}
    ]
    env_ns = {"os": os}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env_ns)
    ns = {
        "os": SimpleNamespace(environ=os.environ if environ is None else environ),
        "henvs": SimpleNamespace(custom_ops_enabled=env_ns["custom_ops_enabled"]),
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    assert len(nodes) == len(names)
    # Include the existing width-consistency check, not just its input flags.
    if "_HOT_CONTEXT_EAGER_FALLBACK" in names:
        nodes += [
            n for n in tree.body
            if isinstance(n, ast.If)
            and any(
                isinstance(t, ast.Name) and t.id == "_HOT_CONTEXT_EAGER_FALLBACK"
                for t in ast.walk(n.test)
            )
        ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("master", ["0", "false", "1", "true"])
def test_expert_placement_is_disabled_by_custom_ops_master(monkeypatch, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_STEP5_EXPERT_PLACEMENT", "/external/placement.pt")
    policy = _policy(MODEL, {"EXPERT_PLACEMENT"})
    assert policy["EXPERT_PLACEMENT"] == (
        "/external/placement.pt" if master in ("1", "true") else ""
    )


@pytest.mark.parametrize("master", ["0", "false", "1", "true"])
def test_dsa_performance_widths_are_disabled_by_custom_ops_master(monkeypatch, master):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_STEP4_DSA_Q_CHUNK", "8192")
    monkeypatch.setenv("VLLM_STEP4_DSA_GRAPH_REGIONS", "1024")
    monkeypatch.setenv("HCU_STEP5_CONTEXT_GRAPH_REGIONS", "1024")
    names = {"INDEXER_Q_CHUNK", "GRAPH_REGIONS_OVERRIDE", "_HOT_CONTEXT_EAGER_FALLBACK"}
    policy = _policy(DSA, names)
    assert tuple(policy[n] for n in (
        "INDEXER_Q_CHUNK", "GRAPH_REGIONS_OVERRIDE", "_HOT_CONTEXT_EAGER_FALLBACK"
    )) == ((8192, 1024, True) if master in ("1", "true") else (0, 0, False))


@pytest.mark.parametrize(
    "path,names",
    [
        (MODEL, {"EXPERT_PLACEMENT"}),
        (DSA, {"INDEXER_Q_CHUNK", "GRAPH_REGIONS_OVERRIDE", "_HOT_CONTEXT_EAGER_FALLBACK"}),
    ],
)
def test_master_off_does_not_read_disabled_optimization_settings(monkeypatch, path, names):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")

    class ForbiddenEnvironment:
        def get(self, *args):
            pytest.fail("master-off read a Step5 optimization setting")

        def __getitem__(self, key):
            pytest.fail("master-off read a Step5 optimization setting")

    _policy(path, names, environ=ForbiddenEnvironment())


def test_enabled_hot_context_still_rejects_inconsistent_widths(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_STEP4_DSA_GRAPH_REGIONS", "1024")
    monkeypatch.setenv("HCU_STEP5_CONTEXT_GRAPH_REGIONS", "2048")
    with pytest.raises(ValueError, match="hot context width must match"):
        _policy(DSA, {"GRAPH_REGIONS_OVERRIDE", "_HOT_CONTEXT_EAGER_FALLBACK"})
