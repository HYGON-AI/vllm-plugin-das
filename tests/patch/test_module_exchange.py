# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import vllm_hcu.patch.import_coordinator as coordinator_module
from vllm_hcu.patch.import_coordinator import (
    ExactImportCoordinator,
    LateModuleReplacementError,
)
from vllm_hcu.patch.module_exchange import (
    _validate_hcu_replacement_path,
    module_exchange_names,
    register_all_module_exchanges,
    register_modular_kernel_exchange,
)
from vllm_hcu.patch.runtime_state import PatchRegistry, PatchStatus


def _remove_exchange_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    for canonical, replacement in module_exchange_names():
        monkeypatch.delitem(sys.modules, canonical, raising=False)
        monkeypatch.delitem(sys.modules, replacement, raising=False)


def test_all_exchange_registration_is_exact_lazy_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    _remove_exchange_targets(monkeypatch)
    original_import = builtins.__import__
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)
    replacements_before = {
        replacement
        for _, replacement in module_exchange_names()
        if replacement in sys.modules
    }

    first = register_all_module_exchanges(coordinator)
    second = register_all_module_exchanges(coordinator)

    assert len(first) == len(second) == 11
    assert len(coordinator.registrations()) == 11
    assert all(item.status == PatchStatus.ARMED.value for item in first)
    assert builtins.__import__ is original_import
    assert replacements_before == set()
    assert not any(
        replacement in sys.modules for _, replacement in module_exchange_names()
    )

    # Exact lookup only: neither ancestors, descendants, nor unrelated names
    # produce a coordinator spec or resolve any replacement.
    assert coordinator.find_spec("vllm.model_executor.layers", None) is None
    assert coordinator.find_spec(
        "vllm.model_executor.layers.linear.unrelated_child", None
    ) is None
    assert coordinator.find_spec("unrelated_hcu_module", None) is None


def test_exchange_inventory_arms_dependencies_before_canonical_consumers():
    order = {
        canonical: index
        for index, (canonical, _) in enumerate(module_exchange_names())
    }
    dependencies = {
        "vllm.model_executor.layers.deepseek_compressor": {
            "vllm.model_executor.layers.linear",
            "vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache",
        },
        "vllm.model_executor.layers.deepseek_v4_attention": {
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.layers.sparse_attn_indexer",
            "vllm.model_executor.layers.deepseek_compressor",
            "vllm.v1.attention.backends.mla.sparse_swa",
        },
        "vllm.model_executor.layers.fused_moe.deep_gemm_utils": {
            "vllm.model_executor.layers.fused_moe.modular_kernel",
        },
        "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe": {
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
    }
    for consumer, required in dependencies.items():
        assert all(order[dependency] < order[consumer] for dependency in required)

    assert "vllm.model_executor.parameter" not in order
    assert order["vllm.model_executor.layers.linear"] < order[
        "vllm.model_executor.layers.deepseek_compressor"
    ]


def test_replacement_consumers_use_canonical_aliases():
    repo = Path(__file__).resolve().parents[2]
    contracts = {
        "vllm_hcu/model_executor/layers/deepseek_v4_attention.py": {
            "vllm.model_executor.layers.sparse_attn_indexer",
            "vllm.model_executor.layers.deepseek_compressor",
            "vllm.v1.attention.backends.mla.sparse_swa",
        },
        "vllm_hcu/model_executor/layers/deepseek_compressor.py": {
            "vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache",
        },
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py": {
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "dpsk_v4_deep_gemm_moe.py": {
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
        "vllm_hcu/models/deepseek_v2.py": {
            "vllm.model_executor.layers.sparse_attn_indexer",
        },
        "vllm_hcu/models/deepseek_v4.py": {
            "vllm.model_executor.layers.deepseek_v4_attention",
            "vllm.model_executor.layers.mhc",
        },
    }
    replacement_by_canonical = dict(module_exchange_names())
    for relative, expected in contracts.items():
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert expected <= imported
        assert not {
            replacement_by_canonical[canonical]
            for canonical in expected
        } & imported


def test_replacement_paths_are_validated_without_importing():
    for _, replacement in module_exchange_names():
        _validate_hcu_replacement_path(replacement)
    with pytest.raises(ModuleNotFoundError, match="has no source"):
        _validate_hcu_replacement_path("vllm_hcu.missing.stage3_module")
    with pytest.raises(ValueError, match="absolute vllm_hcu"):
        _validate_hcu_replacement_path("other_backend.module")


class _ExplodingLoader(importlib.abc.Loader):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def create_module(self, spec):
        self.events.append("official-create")
        raise AssertionError("official modular_kernel must never execute")

    def exec_module(self, module):
        self.events.append("official-exec")
        raise AssertionError("official modular_kernel must never execute")


class _OfficialFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, events: list[str]) -> None:
        self.target = target
        self.events = events

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        self.events.append("official-find")
        return importlib.util.spec_from_loader(
            fullname, _ExplodingLoader(self.events), origin="official-test-module"
        )


def _fake_package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = []
    package.__spec__ = importlib.util.spec_from_loader(
        name, loader=None, is_package=True
    )
    return package


def test_modular_kernel_canonical_import_never_executes_official_module(
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, replacement_name = module_exchange_names()[0]
    assert canonical == "vllm.model_executor.layers.fused_moe.modular_kernel"
    for package_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ):
        monkeypatch.setitem(sys.modules, package_name, _fake_package(package_name))
    monkeypatch.delitem(sys.modules, canonical, raising=False)
    monkeypatch.delitem(sys.modules, replacement_name, raising=False)

    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)
    register_modular_kernel_exchange(coordinator)
    coordinator.install()
    events: list[str] = []
    official_finder = _OfficialFinder(canonical, events)
    sys.meta_path.insert(1, official_finder)

    replacement = ModuleType(replacement_name)
    replacement_imports: list[str] = []
    original_import_module = coordinator_module.importlib.import_module

    def import_replacement(name: str, package: str | None = None):
        if name == replacement_name:
            replacement_imports.append(name)
            return replacement
        return original_import_module(name, package)

    monkeypatch.setattr(
        coordinator_module.importlib, "import_module", import_replacement
    )
    original_builtin_import = builtins.__import__
    try:
        loaded = builtins.__import__(canonical, fromlist=["*"])
    finally:
        while official_finder in sys.meta_path:
            sys.meta_path.remove(official_finder)
        coordinator.reset_for_tests()

    assert loaded is replacement
    assert replacement_imports == [replacement_name]
    assert events == []
    assert builtins.__import__ is original_builtin_import


def test_modular_kernel_strict_late_policy_keeps_official_and_hcu_exclusive(
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, replacement = module_exchange_names()[0]
    official_module = ModuleType(canonical)
    monkeypatch.setitem(sys.modules, canonical, official_module)
    monkeypatch.delitem(sys.modules, replacement, raising=False)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    with pytest.raises(LateModuleReplacementError, match="already imported"):
        register_modular_kernel_exchange(coordinator)

    assert sys.modules[canonical] is official_module
    assert replacement not in sys.modules
    record = registry.get("module_exchange.modular_kernel")
    assert record is not None and record.status is PatchStatus.FAILED
