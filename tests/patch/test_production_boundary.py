# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

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


boundary = _load_tool_module("check_production_boundary")


def test_scanner_rejects_each_production_only_pattern(tmp_path: Path) -> None:
    package = tmp_path / "vllm_hcu"
    package.mkdir()
    (package / "adapter_v024.py").write_text(
        'SEGMENT_TEST_IDS = {"SP-V024-0001": "test_x"}\n'
        '_hcu_v024_applied = True\n',
        encoding="utf-8",
    )

    scanned, violations = boundary.scan_production_package(package)

    assert scanned == 1
    assert {item.kind for item in violations} == {
        "legacy_audit_field",
        "legacy_segment_id",
        "versioned_runtime_marker",
        "versioned_runtime_module",
    }


def test_repository_production_boundary_is_clean() -> None:
    scanned, violations = boundary.scan_production_package(REPOSITORY / "vllm_hcu")

    assert scanned > 0
    assert violations == []
