# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Contract tests for the distribution package boundary."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from setuptools import find_namespace_packages
import torch.utils.cpp_extension as torch_cpp_extension


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
        # The shared HCU workspace can be owned by a mounted host uid.
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
    assert f"0.28.0+das.{sha}.dtk2604" in result.stdout
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


def test_setup_sanitizes_hcu_compiler_command_output() -> None:
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    sanitizer = functions["_sanitize_hcu_build_output"]
    sanitizer_source = ast.unparse(sanitizer)
    assert "_HIP_PLATFORM_DEFINES" in sanitizer_source
    assert "_REDACTED_HIP_PLATFORM_DEFINE" in sanitizer_source

    sanitizer_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {"_HIP_PLATFORM_DEFINES", "_REDACTED_HIP_PLATFORM_DEFINE"}
        )
        or node is sanitizer
    ]
    sanitizer_namespace: dict[str, object] = {
        "torch_cpp_extension": torch_cpp_extension,
    }
    exec(
        compile(
            ast.Module(body=sanitizer_nodes, type_ignores=[]),
            filename="setup.py",
            mode="exec",
        ),
        sanitizer_namespace,
    )
    platform_defines = sanitizer_namespace["_HIP_PLATFORM_DEFINES"]
    sanitize = sanitizer_namespace["_sanitize_hcu_build_output"]
    assert isinstance(platform_defines, tuple)
    assert platform_defines
    assert all(isinstance(flag, str) for flag in platform_defines)
    assert callable(sanitize)
    compiler_output = f"compiler {' '.join(platform_defines)} failed"
    sanitized = sanitize(compiler_output)
    if any(flag in sanitized for flag in platform_defines):
        raise AssertionError("HIP platform compiler define was not sanitized")
    assert "<hcu-platform-define>" in sanitized

    ninja_boundary = functions["_sanitized_ninja_build"]
    boundary_source = ast.unparse(ninja_boundary)
    assert "original_run_ninja_build(build_directory, False" in boundary_source
    assert "_sanitize_hcu_build_output(str(error))" in boundary_source

    build_ext = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CustomBuildExt"
    )
    run_method = next(
        node
        for node in build_ext.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sanitized_ninja_build"
        for node in ast.walk(run_method)
    )

    extension_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CUDAExtension"
    )
    extra_compile_args = next(
        keyword.value
        for keyword in extension_call.keywords
        if keyword.arg == "extra_compile_args"
    )
    compile_args = ast.literal_eval(extra_compile_args)
    assert not any(
        argument.startswith("-D__HIP_PLATFORM_")
        for arguments in compile_args.values()
        for argument in arguments
    )
