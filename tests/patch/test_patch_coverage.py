# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Repository-wide structural coverage gates for every patch adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_tool_module(name: str):
    path = REPOSITORY / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"plugin_tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


coverage_gate = _load_tool_module("check_patch_test_coverage")


def test_every_patch_module_has_contract_and_direct_test_reference() -> None:
    audit = coverage_gate.audit_repository(REPOSITORY)

    assert audit.coordinator_helpers == (
        "vllm_hcu.patch.platform.core_fix.patch_hcu_config",
    )
    assert audit.patch_file_count == (
        audit.adapter_count + len(audit.coordinator_helpers)
    )
    assert audit.adapter_count > 0
    assert audit.missing_contract == {}
    assert audit.untested_modules == ()


def test_coverage_gate_rejects_an_uncontracted_untested_patch(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    patch_root = repository / "vllm_hcu" / "patch" / "worker" / "op_opt"
    patch_root.mkdir(parents=True)
    test_root = repository / "tests"
    test_root.mkdir()
    (patch_root / "patch_new_backend.py").write_text(
        'PATCH_ID = "worker.op_opt.new_backend"\n',
        encoding="utf-8",
    )
    (test_root / "test_something_else.py").write_text(
        "def test_something_else():\n    assert True\n",
        encoding="utf-8",
    )

    audit = coverage_gate.audit_repository(repository)

    assert audit.missing_contract == {
        "vllm_hcu.patch.worker.op_opt.patch_new_backend": (
            "TARGET_MODULE",
            "apply_to_module",
        )
    }
    assert audit.untested_modules == (
        "vllm_hcu.patch.worker.op_opt.patch_new_backend",
    )
