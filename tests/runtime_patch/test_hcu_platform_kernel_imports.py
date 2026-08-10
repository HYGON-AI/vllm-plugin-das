# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hcu_platform_imports_stable_libtorch_kernels() -> None:
    source = (ROOT / "vllm_hcu/platforms/hcu.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    hcu_platform = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "HCUPlatform"
    )
    method = next(
        node
        for node in hcu_platform.body
        if isinstance(node, ast.FunctionDef) and node.name == "import_kernels"
    )
    imported_modules = {
        alias.name
        for node in ast.walk(method)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "vllm._C_stable_libtorch" in imported_modules
