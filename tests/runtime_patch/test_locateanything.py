# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for package-owned LocateAnything loading and isolation."""
from pathlib import Path
import ast
import importlib.util
import hashlib
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter():
    return load_file("locate_patch_test", "vllm_hcu/patch/locateanything.py")


def test_slow_does_not_exchange_shared_modules(monkeypatch, adapter):
    monkeypatch.delenv("VLLM_HCU_LOCATEANYTHING_MODE", raising=False)
    class Coordinator:
        def register_replacement(self, *args, **kwargs):
            pytest.fail("Slow must keep the shared vLLM implementations")
    adapter.register_locateanything_exchanges(Coordinator())


def test_invalid_mode_is_rejected(monkeypatch, adapter):
    monkeypatch.setenv("VLLM_HCU_LOCATEANYTHING_MODE", "fas")
    with pytest.raises(ValueError, match="slow or fast"):
        adapter.locateanything_mode()


def test_fast_registers_all_seven_cold_imports(monkeypatch, adapter):
    monkeypatch.setenv("VLLM_HCU_LOCATEANYTHING_MODE", "fast")
    monkeypatch.setattr(adapter, "validate_fast_sources", lambda contract: None)
    calls = []
    class Coordinator:
        def register_replacement(self, *args, **kwargs):
            calls.append((args, kwargs))
    adapter.register_locateanything_exchanges(Coordinator())
    assert len(calls) == 7
    names = {args[1] for args, _ in calls}
    assert "vllm.v1.core.sched.scheduler" in names
    assert "vllm.v1.attention.ops.triton_unified_attention" in names
    for args, kwargs in calls:
        assert callable(args[2])
        assert "late_policy" not in kwargs  # coordinator's fail-closed default
    for spec in adapter.fast_source_contract().values():
        assert (ROOT / (spec['module'].replace(".", "/") + ".py")).is_file()


def test_source_drift_fails_before_registration(tmp_path, adapter):
    path = tmp_path / "source.py"; path.write_bytes(b"qualified source")
    contract = {"source.py": {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    adapter.validate_fast_sources(contract, tmp_path)
    path.write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="source contract mismatch"):
        adapter.validate_fast_sources(contract, tmp_path)


def test_fast_registration_is_reentrant():
    # The suite can already have imported the GPU runner. Cold registration is
    # a startup contract: use a fresh interpreter and leave parent modules alone.
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/runtime_patch/fixtures/locateanything_cold_import.py")],
        cwd=ROOT, env={**os.environ, "VLLM_PLUGINS": "__disabled__", "VLLM_HCU_LOCATEANYTHING_MODE": "slow",
                       "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cold registration, reentry and late-import rejection passed" in result.stdout


def test_replacement_preserves_globals_and_exact_callback_name(monkeypatch, adapter):
    import sys
    owner = "vllm_hcu.v1.locateanything.contract_fixture"
    canonical = "vllm.locateanything_contract_fixture"
    module = load_file(owner, "tests/runtime_patch/fixtures/locateanything_globals.py")
    module.__file__ = "/installed/vllm_hcu/fixture.py"
    monkeypatch.setitem(sys.modules, owner, module)
    factory = adapter._canonical_factory(canonical, owner)
    assert factory is adapter._canonical_factory(canonical, owner)
    actual = factory()
    common = load_file("locate_callback_common", "vllm_hcu/patch/worker/op_opt/_common.py")
    assert common.load_exact_module(canonical, actual) is module
    actual.value = 7
    assert actual.read_value() == 7
    assert actual.__file__ == "/installed/vllm_hcu/fixture.py"
    assert factory() is actual


@pytest.mark.parametrize("mode,sequences,last", [("slow",16,16),("fast",16,192),("fast",64,768)])
def test_launch_graph_capacity_and_packaged_template(mode, sequences, last):
    serve = load_file("locate_serve_test", "vllm_hcu/models/locateanything/serve.py")
    cmd = serve.build_command(mode, "/model", sequences, [])
    assert "--enforce-eager" not in cmd
    assert cmd[cmd.index("--cudagraph-metrics")-1] == str(last)
    template = Path(cmd[cmd.index("--chat-template")+1])
    assert template.is_relative_to(ROOT / "vllm_hcu") and template.is_file()
    assert "--no-enable-chunked-prefill" in cmd if mode == "fast" else "--no-enable-chunked-prefill" not in cmd


def external_adapter_imports(source):
    forbidden = {"fast_core", "fast_warmup", "request_contract", "teacher_compare", "locateanything_vllm_native"}
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    return [name for name in imports if name.split(".")[0] in forbidden]


@pytest.mark.parametrize("source", [
    "import fast_core", "import fast_core as alias", "import os, fast_core.helpers as alias",
    "from fast_core import value", "from fast_core.helpers import value",
    "def nested():\n    import fast_warmup",
])
def test_external_dependency_guard_covers_absolute_import_forms(source):
    assert external_adapter_imports(source)


@pytest.mark.parametrize("source", [
    "import os, vllm_hcu.v1.locateanything.fast_warmup",
    "from .fast_warmup import warmup", "from vllm_hcu.v1.locateanything import fast_warmup",
])
def test_external_dependency_guard_allows_package_owned_imports(source):
    assert not external_adapter_imports(source)


def test_fast_imports_have_no_external_adapter_dependency():
    runtime = ROOT / "vllm_hcu/v1/locateanything"
    for p in runtime.glob("*.py"):
        assert not external_adapter_imports(p.read_text(encoding="utf-8")), p
