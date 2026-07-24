"""Contract tests for the distribution package boundary."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from setuptools import find_namespace_packages


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INCLUDE = ["vllm_hcu", "vllm_hcu.*"]


def _setup_package_include() -> list[str]:
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        packages = next(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "packages"
        )
        assert isinstance(packages, ast.Call)
        assert isinstance(packages.func, ast.Name)
        assert packages.func.id == "find_namespace_packages"
        include = next(
            keyword.value
            for keyword in packages.keywords
            if keyword.arg == "include"
        )
        return ast.literal_eval(include)
    raise AssertionError("setup(packages=...) was not found")


def test_setup_discovers_only_vllm_hcu_namespace() -> None:
    include = _setup_package_include()
    assert include == EXPECTED_INCLUDE

    discovered = find_namespace_packages(where=REPO_ROOT, include=include)
    assert "vllm_hcu" in discovered
    assert discovered
    assert all(
        package == "vllm_hcu" or package.startswith("vllm_hcu.")
        for package in discovered
    )
    assert not any(
        package == excluded or package.startswith(f"{excluded}.")
        for package in discovered
        for excluded in ("tests", "docs", "examples", "tools")
    )


def test_setup_version_query_keeps_provenance_without_rewriting_source(
    tmp_path: Path,
) -> None:
    version_file = REPO_ROOT / "vllm_hcu" / "version.py"
    before = hashlib.sha256(version_file.read_bytes()).hexdigest()
    rocm_path = tmp_path / "dtk"
    rocm_info = rocm_path / ".info"
    rocm_info.mkdir(parents=True)
    (rocm_info / "rocm_version").write_text("26.0.4\n", encoding="utf-8")
    sha = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT}",
            "rev-parse",
            "--short=7",
            "HEAD",
        ],
        # The shared DCU workspace can be owned by a mounted host uid.
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    env = dict(os.environ)
    env["ADD_GIT_VERSION"] = "1"
    env["ROCM_PATH"] = str(rocm_path)

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"0.25.1+das.{sha}.dtk2604" in result.stdout
    assert hashlib.sha256(version_file.read_bytes()).hexdigest() == before


def test_setup_has_no_version_source_or_global_git_config_mutation() -> None:
    source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "write_version_file" not in function_names
    assert '"--global"' not in source
    assert 'ROOT / "vllm_hcu" / "version.py"' not in source
