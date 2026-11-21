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
    result.model.loaded_parts = {}
    calls = []
    result.model.verify_loaded_parts = lambda: calls.append("shards")
    monkeypatch.setattr(DefaultModelLoader, "track_weights_loading",
        lambda self, *args, **kwargs: calls.append("core"))
    return result, calls


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
