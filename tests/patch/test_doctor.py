# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from vllm_hcu import doctor
from vllm_hcu import compatibility
from vllm_hcu.post_install import apply_post_install_patches


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_doctor_detects_markers_and_hcu_symlinks_without_mutating(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    clean = package / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    patched = package / "patched.py"
    patched.write_text("PATCHED_TEST = True\n", encoding="utf-8")
    hcu_target = tmp_path / "vllm_hcu" / "implementation.py"
    hcu_target.parent.mkdir()
    hcu_target.write_text("value = 2\n", encoding="utf-8")
    link = package / "linked.py"
    link.symlink_to(hcu_target)
    before = {path: path.lstat() for path in (clean, patched, hcu_target, link)}

    checks = {item.check: item for item in doctor._source_integrity_checks(package)}

    assert not checks["no_source_patch_markers"].ok
    assert "patched.py" in checks["no_source_patch_markers"].detail
    assert not checks["no_hcu_source_symlinks"].ok
    assert "linked.py" in checks["no_hcu_source_symlinks"].detail
    assert {path: path.lstat() for path in before} == before


def test_doctor_reuses_runtime_compatibility_check_name_and_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Distribution:
        version = "0.22.0"

        @staticmethod
        def locate_file(path: str) -> Path:
            assert path == ""
            return tmp_path

    monkeypatch.setattr(
        compatibility.importlib_metadata,
        "distribution",
        lambda name: Distribution(),
    )

    checks = {
        item.check: item
        for item in doctor.collect_diagnostics(arm_platform=False)
    }

    assert "vllm_v021" not in checks
    assert not checks["vllm_compatible"].ok
    assert "expected=0.21.x" in checks["vllm_compatible"].detail
    assert "actual='0.22.0'" in checks["vllm_compatible"].detail


def test_legacy_apply_command_is_read_only(capsys):
    modular_kernel = Path(
        "/usr/local/lib/python3.10/dist-packages/vllm/"
        "model_executor/layers/fused_moe/modular_kernel.py"
    )
    before_hash = _sha256(modular_kernel) if modular_kernel.is_file() else None
    result = apply_post_install_patches(["--no-arm", "--json"])
    after_hash = _sha256(modular_kernel) if modular_kernel.is_file() else None

    # The result still reflects the surrounding vLLM installation, while the
    # compatibility command's safety property is independent of that result.
    assert result in (0, 1)
    assert before_hash == after_hash
    assert "read-only" in capsys.readouterr().out


def test_post_install_contains_no_filesystem_mutation_calls():
    source = (REPO_ROOT / "vllm_hcu/post_install.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "remove", "replace", "symlink", "symlink_to", "unlink"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    calls.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert forbidden.isdisjoint(calls)


def test_setup_exposes_both_read_only_diagnostic_commands():
    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "vllm-hcu-apply-patches = vllm_hcu.post_install:main" in setup_source
    assert "vllm-hcu-doctor = vllm_hcu.doctor:main" in setup_source


def test_repository_contains_no_legacy_source_patch_path():
    package_root = REPO_ROOT / "vllm_hcu"

    assert not (package_root / "patches").exists()
    assert not (package_root / "patch_utils.py").exists()
    assert list(package_root.rglob("vllm*.patch.py")) == []
