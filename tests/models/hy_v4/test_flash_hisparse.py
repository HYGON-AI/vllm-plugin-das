# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HiSparse CPU admission/model-wiring tests, not HCU numerical qualification.

Adapter/native-dispatch and framework objects are explicit doubles. These tests
run production validation and the window builder override; they do not pretend
that four registered schemas prove working GPU kernels.
"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

FLASH = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash"
OPS = (
    "hisparse_resolve_residency", "hisparse_invalidate_written_slots",
    "hisparse_gather_plan", "hisparse_gather_compact",
)


def config(connector=None, enabled=False, eager=True, spec=None):
    return NS(
        kv_transfer_config=None if connector is None else NS(kv_connector=connector),
        attention_config=NS(hisparse_config=NS() if enabled else None),
        model_config=NS(enforce_eager=eager), speculative_config=spec,
    )


@pytest.fixture
def lane(monkeypatch):
    package = "_hy4_hisparse_test"
    runtime = ModuleType(package + ".runtime")
    runtime.is_hcu_platform = lambda: True
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    spec = importlib.util.spec_from_file_location(package + ".hisparse", FLASH / "hisparse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    native = NS(**{name: object() for name in OPS})
    torch = ModuleType("torch")
    torch.ops = NS(_C_cache_ops=native)
    torch._C = NS(_dispatch_has_kernel_for_dispatch_key=lambda name, key: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    adapter = ModuleType("hcu_adapter.kernels.hisparse")
    adapter.ensure_hisparse_ops = lambda: calls.append("load")
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    return module, calls, native, torch


def test_disabled_does_not_load_adapter_or_require_ops(lane):
    module, calls, native, _ = lane
    for name in OPS:
        delattr(native, name)
    assert module.validate_flash_hisparse(config()) is False
    assert calls == []


@pytest.mark.parametrize("connector", ["NixlConnector", "MooncakeConnector", "MultiConnector"])
def test_reject_other_connectors_before_loading(lane, connector):
    module, calls, *_ = lane
    with pytest.raises(ValueError, match="only the HiSparseConnector"):
        module.validate_flash_hisparse(config(connector, enabled=True))
    assert calls == []


def test_require_connector_for_direct_hisparse_config(lane):
    module, calls, *_ = lane
    with pytest.raises(ValueError, match="requires HiSparseConnector"):
        module.validate_flash_hisparse(config(enabled=True))
    assert calls == []


def test_require_resolved_hisparse_config(lane):
    module, calls, *_ = lane
    with pytest.raises(ValueError, match="resolve"):
        module.validate_flash_hisparse(config("HiSparseConnector"))
    assert calls == []


@pytest.mark.parametrize("method", ["dfly", "eagle", "ngram"])
def test_reject_speculation_until_shared_hot_owner_abi_is_ported(lane, method):
    module, calls, *_ = lane
    with pytest.raises(NotImplementedError, match="shared-hot-cache ABI"):
        module.validate_flash_hisparse(config("HiSparseConnector", True, spec=NS(method=method)))
    assert calls == []


def test_reject_unqualified_graph_lane(lane):
    module, calls, *_ = lane
    with pytest.raises(NotImplementedError, match="enforce_eager"):
        module.validate_flash_hisparse(config("HiSparseConnector", True, eager=False))
    assert calls == []


def test_explicit_target_only_lane_loads_adapter(lane):
    module, calls, *_ = lane
    assert module.validate_flash_hisparse(config("HiSparseConnector", True)) is True
    assert calls == ["load"]


def test_cuda_uses_core_ops_without_adapter(lane, monkeypatch):
    module, calls, *_ = lane
    monkeypatch.setattr(sys.modules[module.__package__ + ".runtime"], "is_hcu_platform", lambda: False)
    assert module.validate_flash_hisparse(config("HiSparseConnector", True)) is True
    assert calls == []


@pytest.mark.parametrize("missing", OPS)
def test_missing_native_operator_is_not_treated_as_support(lane, missing):
    module, calls, native, _ = lane
    delattr(native, missing)
    with pytest.raises(RuntimeError, match=missing):
        module.validate_flash_hisparse(config("HiSparseConnector", True))
    assert calls == ["load"]


@pytest.mark.parametrize("missing", OPS)
def test_schema_without_gpu_dispatch_is_not_treated_as_support(lane, missing):
    module, _, _, torch = lane
    torch._C._dispatch_has_kernel_for_dispatch_key = lambda name, key: name != f"_C_cache_ops::{missing}"
    with pytest.raises(RuntimeError, match=missing):
        module.validate_flash_hisparse(config("HiSparseConnector", True))


def test_old_adapter_reports_actionable_dependency(lane, monkeypatch):
    module, calls, *_ = lane
    monkeypatch.setitem(sys.modules, "hcu_adapter.kernels.hisparse", None)
    with pytest.raises(RuntimeError, match="native HIP"):
        module.validate_flash_hisparse(config("HiSparseConnector", True))
    assert calls == []


def test_adapter_load_failure_propagates_without_fallback(lane, monkeypatch):
    module, _, *_ = lane
    adapter = sys.modules["hcu_adapter.kernels.hisparse"]
    def fail():
        raise RuntimeError("HIP build unavailable")
    monkeypatch.setattr(adapter, "ensure_hisparse_ops", fail)
    with pytest.raises(RuntimeError, match="HIP build unavailable"):
        module.validate_flash_hisparse(config("HiSparseConnector", True))


@pytest.mark.parametrize("producing", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_window_builder_never_gives_hisparse_ownership(producing, enabled):
    tree = ast.parse((FLASH / "nvidia/swa.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashWindowIndexGroupBuilder")
    calls = []
    class Builder:
        def register_layer(self, producing, config, **kwargs):
            calls.append((producing, config, kwargs))
            return "window-owned-group", 0
    ns = {"SparseMLAIndexGroupBuilder": Builder}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<window-builder>", "exec"), ns)
    builder = ns["FlashWindowIndexGroupBuilder"]()
    cfg = config("HiSparseConnector", enabled)
    assert builder.register_layer(producing, cfg, head_size=576, kv_cache_dtype="bfloat16") == ("window-owned-group", 0)
    assert calls == [(producing, None, {"head_size": 576, "kv_cache_dtype": "bfloat16"})]
    assert (cfg.attention_config.hisparse_config is not None) == enabled


@pytest.mark.parametrize("window", [True, False])
def test_attention_only_installs_window_builder_for_window_layers(window):
    tree = ast.parse((FLASH / "nvidia/attention.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MLAAttention")
    expression = next(k.value for k in call.keywords if k.arg == "index_group_builder")
    buffer = object()
    result = eval(compile(ast.Expression(expression), "<window-wiring>", "eval"), {
        "self": NS(is_window_layer=window), "topk_indices_buffer": buffer,
        "FlashWindowIndexGroupBuilder": lambda value: ("window", value),
    })
    assert result == (("window", buffer) if window else None)


def test_model_calls_hisparse_validator_once_and_keeps_lora_rejection():
    tree = ast.parse((FLASH / "nvidia/model.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_runtime")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "validate_flash_hisparse"]
    assert len(calls) == 1
    assert "HYV4 Flash does not support LoRA" in ast.unparse(fn)


@pytest.mark.parametrize("enabled", [False, True])
def test_window_registration_with_installed_core_builder_cpu(enabled):
    """Run installed Core's real registration/dataclass code and Torch CPU.

    Streams/events are inert doubles; HiSparse cache construction is deliberately
    forbidden, not simulated as a working HCU cache. No framework imports or
    GPU allocations are needed to verify this ownership boundary.
    """
    import __future__
    from dataclasses import dataclass
    torch = pytest.importorskip("torch")

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("requires an installed vLLM source package")
    source = Path(spec.origin).parent / "v1/attention/backends/mla/index_group.py"
    if not source.exists():
        pytest.skip("installed Core has no HiSparse index-group source")
    core = ast.parse(source.read_text())
    names = {"SparseMLAIndexGroup", "SparseMLAIndexGroupBuilder"}
    nodes = [n for n in core.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == 2
    class ForbiddenHiSparse:
        def __init__(self, **kwargs):
            raise AssertionError("window registered with HiSparse ownership")
    ns = {
        "__name__": __name__, "torch": torch, "dataclass": dataclass,
        "HiSparseMLAIndexGroup": ForbiddenHiSparse,
        "_create_side_stream": lambda device: None, "_create_event": lambda: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag), ns)
    own = ast.parse((FLASH / "nvidia/swa.py").read_text())
    node = next(n for n in own.body if isinstance(n, ast.ClassDef) and n.name == "FlashWindowIndexGroupBuilder")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<window-builder>", "exec"), ns)
    indices = torch.zeros((4, 8), dtype=torch.int32)
    builder = ns["FlashWindowIndexGroupBuilder"](indices)
    cfg = config("HiSparseConnector", enabled)
    group, index = builder.register_layer(True, cfg, head_size=576, kv_cache_dtype="bfloat16")
    assert type(group) is ns["SparseMLAIndexGroup"]
    assert index == 0 and group.num_layers == 1
    assert group.logical_topk_indices is indices
    assert group.physical_topk_indices.shape == (5, 8)
    assert group.valid_topk_counts.shape == (5,)
    same, index = builder.register_layer(False, cfg, head_size=576, kv_cache_dtype="bfloat16")
    assert same is group and index == 1 and group.num_layers == 2
    if enabled:
        with pytest.raises(AssertionError, match="HiSparse ownership"):
            ns["SparseMLAIndexGroupBuilder"](indices).register_layer(True, cfg)
