# SPDX-License-Identifier: Apache-2.0
"""Explicit Flash loading audit, independent of Core tracker strictness.

Executes the production context manager with real Torch parameters. The final
Core tracker and model shard auditor are recording doubles, not model loaders.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash/nvidia/model.py"
PACKED = [f"w{projection}_weight_{suffix}" for projection in ("13", "2")
          for suffix in ("packed", "scale", "shape")]


def model(monkeypatch):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.config import LoadConfig
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4FlashForCausalLM")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "weight_loading_session")
    namespace = {"contextmanager": contextmanager}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    result = type("FlashAuditShell", (torch.nn.Module,), {
        "weight_loading_session": namespace["weight_loading_session"]})()
    result._weight_loading_depth = 0
    result._seen_weights = set()
    result._loaded_weights = set()
    result._load_config = LoadConfig()
    result.model = torch.nn.Module()
    result.model.experts = torch.nn.Module()
    for name in PACKED:
        result.model.experts.register_parameter(name,
            torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    result.model.norm = torch.nn.Module()
    result.model.norm.weight = torch.nn.Parameter(torch.ones(1))
    result.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in result.model.layers:
        layer.mlp = torch.nn.Module()
        layer.mlp.expert_bias = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    result.model.loaded_parts = {}
    calls = []
    result.model.verify_loaded_parts = lambda: calls.append("shards")
    monkeypatch.setattr(DefaultModelLoader, "track_weights_loading",
        lambda self, *args, **kwargs: calls.append("core"))
    return result, calls


@pytest.mark.parametrize("layer", [0, 1])
def test_entire_correction_bias_missing_rejects_without_core_tracker(monkeypatch, layer):
    shell, calls = model(monkeypatch)
    omitted = f"model.layers.{layer}.mlp.expert_bias"
    with pytest.raises(ValueError, match=omitted):
        with shell.weight_loading_session():
            shell._loaded_weights.update(name for name, _ in shell.named_parameters() if name != omitted)
    assert calls == []
    assert shell._weight_loading_depth == 0 and not shell._loaded_weights


@pytest.mark.parametrize("checkpoint_leaf", ["expert_bias", "gate.e_score_correction_bias"])
def test_correction_bias_canonical_and_checkpoint_alias_load_completely(monkeypatch, checkpoint_leaf):
    import regex as re
    from types import MethodType
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    shell, calls = model(monkeypatch)
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HYV4FlashModel")
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_EXPERT_WEIGHT" for t in n.targets)]
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("load_weights", "_record_part")]
    namespace = {"re": re, "default_weight_loader": default_weight_loader}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    shell.model._record_part = MethodType(namespace["_record_part"], shell.model)
    values = [torch.arange(3).float() + layer + 1 for layer in range(2)]
    with shell.weight_loading_session():
        shell._loaded_weights.update(name for name, _ in shell.named_parameters() if not name.endswith(".expert_bias"))
        loaded = namespace["load_weights"](shell.model,
            ((f"layers.{layer}.mlp.{checkpoint_leaf}", value) for layer, value in enumerate(values)))
        shell._loaded_weights.update("model." + name for name in loaded)
    assert calls == ["core", "shards"]
    for layer, value in zip(shell.model.layers, values):
        torch.testing.assert_close(layer.mlp.expert_bias, value, rtol=0, atol=0)


@pytest.mark.parametrize("missing", PACKED)
def test_entire_packed_parameter_missing_is_rejected_before_core_tracker(monkeypatch, missing):
    shell, calls = model(monkeypatch)
    omitted = "model.experts." + missing
    with pytest.raises(ValueError, match=missing):
        with shell.weight_loading_session():
            shell._loaded_weights.update(name for name, _ in shell.named_parameters() if name != omitted)
    assert calls == []
    assert shell._weight_loading_depth == 0
    assert not shell._seen_weights and not shell._loaded_weights


def test_complete_nested_stream_is_audited_once_after_outermost_load(monkeypatch):
    shell, calls = model(monkeypatch)
    with shell.weight_loading_session():
        shell._loaded_weights.update(name for name, _ in shell.named_parameters())
        with shell.weight_loading_session():
            assert shell._weight_loading_depth == 2
        assert calls == []
    assert calls == ["core", "shards"]
    assert shell._weight_loading_depth == 0 and not shell._loaded_weights


def test_stream_error_preserves_exception_and_cleans_audit_state(monkeypatch):
    shell, calls = model(monkeypatch)
    with pytest.raises(RuntimeError, match="broken stream"):
        with shell.weight_loading_session():
            shell._seen_weights.add("sentinel")
            raise RuntimeError("broken stream")
    assert calls == [] and shell._weight_loading_depth == 0
    assert not shell._seen_weights and not shell._loaded_weights
