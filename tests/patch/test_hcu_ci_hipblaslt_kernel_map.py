# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CI-only compatibility repair must be exact, atomic and idempotent."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ensure_hipblaslt_kernel_map",
    ROOT / ".github/scripts/hcu_ci/ensure_hipblaslt_kernel_map.py",
)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


@pytest.fixture
def tables(tmp_path, monkeypatch):
    directory = tmp_path / "library_gfx938"
    directory.mkdir()
    original = b"existing_kernel\n"
    names = original + repair.MISSING_KERNEL.encode() + b"\n"
    binary = b"code object symbol: " + repair.MISSING_KERNEL.encode()
    for filename, data, attribute in (
        ("w8a8_customkernel_all.txt", original, "BROKEN_MAP_SHA256"),
        ("FusedMatmulKernelName.txt", names, "FUSED_NAMES_SHA256"),
        ("FusedMatmulKernel_gfx938.co", binary, "FUSED_BINARY_SHA256"),
    ):
        (directory / filename).write_bytes(data)
        monkeypatch.setattr(repair, attribute, hashlib.sha256(data).hexdigest())
    return tmp_path, directory / "w8a8_customkernel_all.txt", original


def test_repairs_only_missing_name_and_is_idempotent(tables):
    root, mapping, original = tables
    mapping.chmod(0o640)
    assert repair.ensure_kernel_map(root)
    assert mapping.read_bytes() == original + repair.MISSING_KERNEL.encode() + b"\n"
    assert mapping.stat().st_mode & 0o777 == 0o640
    assert not repair.ensure_kernel_map(root)
    assert not list(mapping.parent.glob("tmp*"))


def test_other_versions_are_untouched(tables):
    root, mapping, _ = tables
    mapping.write_bytes(b"different release\n")
    assert not repair.ensure_kernel_map(root)
    assert mapping.read_bytes() == b"different release\n"


@pytest.mark.parametrize(
    "filename",
    ["FusedMatmulKernelName.txt", "FusedMatmulKernel_gfx938.co"],
)
def test_rejects_mismatched_companion_files(tables, filename):
    root, mapping, original = tables
    (mapping.parent / filename).write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="unexpected companion"):
        repair.ensure_kernel_map(root)
    assert mapping.read_bytes() == original


def test_rejects_additional_missing_names(tables, monkeypatch):
    root, mapping, original = tables
    names = original + repair.MISSING_KERNEL.encode() + b"\nanother_missing_kernel\n"
    (mapping.parent / "FusedMatmulKernelName.txt").write_bytes(names)
    monkeypatch.setattr(repair, "FUSED_NAMES_SHA256", hashlib.sha256(names).hexdigest())
    with pytest.raises(RuntimeError, match="Cannot safely repair"):
        repair.ensure_kernel_map(root)
    assert mapping.read_bytes() == original


def test_missing_library_is_untouched(tmp_path):
    assert not repair.ensure_kernel_map(tmp_path)


def test_ci_installs_repair_before_hardware_tests():
    script = (ROOT / "scripts/ci/hcu/hcu_ci_install_dependency.sh").read_text()
    assert "ensure_hipblaslt_kernel_map.py" in script
