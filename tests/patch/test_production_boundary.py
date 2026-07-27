# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from pathlib import Path

from tools import check_production_boundary as boundary

REPOSITORY = Path(__file__).resolve().parents[2]


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
