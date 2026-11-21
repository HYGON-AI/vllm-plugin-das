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
    for layer in result.model.layers:
        layer.self_attn = torch.nn.Module()
        layer.self_attn.indexer = torch.nn.Module()
        layer.self_attn.indexer.intra_group_ape = torch.nn.Parameter(torch.empty(2, 128))
        layer.self_attn.learnable_sink_param = torch.nn.Parameter(torch.empty(1))
        layer.engram_layer = torch.nn.Module()
        layer.engram_layer.unified_pre_gamma = torch.nn.Parameter(torch.empty(3))
        layer.engram_layer.unified_per_hc_scales = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(3)) for _ in range(2)])
        layer.engram_layer.value_proj = torch.nn.Linear(2, 3)
        layer.engram_layer.key_projs = torch.nn.ModuleList(
            [torch.nn.Linear(2, 3) for _ in range(2)])
        for name in ("hc_attn_layer", "hc_mlp_layer"):
            module = torch.nn.Module()
            module.hc_pre = torch.nn.Module()
            module.hc_pre.hc_scale = torch.nn.Parameter(torch.empty(2))
            module.hc_pre.hc_base = torch.nn.Parameter(torch.empty(4))
            setattr(layer, name, module)
    result.model.hc_head = torch.nn.Module()
    result.model.hc_head.hc_head_base = torch.nn.Parameter(torch.empty(2))
    result.model.hc_head.hc_head_scale = torch.nn.Parameter(torch.empty(1))
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


NON_WEIGHT_LEAVES = (
    "self_attn.indexer.intra_group_ape", "self_attn.learnable_sink_param",
    "engram_layer.unified_pre_gamma", "engram_layer.unified_per_hc_scales.0",
    "engram_layer.unified_per_hc_scales.1", "engram_layer.value_proj.bias",
    "engram_layer.key_projs.0.bias", "engram_layer.key_projs.1.bias",
    "hc_attn_layer.hc_pre.hc_scale", "hc_attn_layer.hc_pre.hc_base",
    "hc_mlp_layer.hc_pre.hc_scale", "hc_mlp_layer.hc_pre.hc_base",
)
MANDATORY_NON_WEIGHTS = tuple(
    f"model.layers.{layer}.{leaf}" for layer in range(2) for leaf in NON_WEIGHT_LEAVES
) + ("model.hc_head.hc_head_base", "model.hc_head.hc_head_scale")


@pytest.mark.parametrize("omitted", MANDATORY_NON_WEIGHTS)
def test_entire_learned_non_weight_parameter_missing_rejects(monkeypatch, omitted):
    shell, calls = model(monkeypatch)
    assert omitted in dict(shell.named_parameters())
    with pytest.raises(ValueError, match=omitted):
        with shell.weight_loading_session():
            shell._loaded_weights.update(name for name, _ in shell.named_parameters() if name != omitted)
    assert calls == []
    assert shell._weight_loading_depth == 0 and not shell._loaded_weights


def test_mandatory_non_weights_load_through_actual_assignment(monkeypatch):
    import regex as re
    from types import MethodType
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    shell, calls = model(monkeypatch)
    shell.model.config = SimpleNamespace(engram_layer_ids=(1, 2))
    shell.model.engram_hash = object()
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HYV4FlashModel")
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_EXPERT_WEIGHT" for t in n.targets)]
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("load_weights", "_record_part")]
    namespace = {"re": re, "default_weight_loader": default_weight_loader}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    shell.model._record_part = MethodType(namespace["_record_part"], shell.model)
    parameters = dict(shell.named_parameters())
    values = {name: torch.full_like(parameters[name], i + 1) for i, name in enumerate(MANDATORY_NON_WEIGHTS)}
    with shell.weight_loading_session():
        shell._loaded_weights.update(name for name in parameters if name not in values)
        loaded = namespace["load_weights"](shell.model,
            ((name.removeprefix("model."), value) for name, value in values.items()))
        assert loaded == {name.removeprefix("model.") for name in values}
        shell._loaded_weights.update("model." + name for name in loaded)
    assert calls == ["core", "shards"]
    for name, value in values.items():
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)


def test_disabled_hc_constructor_has_no_mandatory_hc_parameters(monkeypatch):
    shell, calls = model(monkeypatch)
    source = SOURCE.parents[2] / "hy_v4/hc.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4HCLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    # Keep the real conditional constructor and Python's zero-argument super.
    body = ast.parse("class DisabledHC(torch.nn.Module):\n    pass\n").body[0]
    body.body = [init]
    nodes = [ast.ImportFrom(module="__future__",
                           names=[ast.alias(name="annotations")], level=0), body]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source), "exec"), namespace)
    for layer in shell.model.layers:
        for name in ("hc_attn_layer", "hc_mlp_layer"):
            module = namespace["DisabledHC"](SimpleNamespace(enable_ihc=False), 0)
            assert not list(module.named_parameters())
            setattr(layer, name, module)
    del shell.model.hc_head
    with shell.weight_loading_session():
        shell._loaded_weights.update(name for name, _ in shell.named_parameters())
    assert calls == ["core", "shards"]
