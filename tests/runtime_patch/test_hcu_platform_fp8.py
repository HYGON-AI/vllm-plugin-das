# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("gfx938", True),
        ("gfx946", True),
        ("gfx936", False),
        ("gfx92a", False),
        ("gfx950", False),
        ("", False),
    ],
)
def test_hcu_fp8_capability_is_architecture_specific(arch, expected):
    """Execute the platform query without initializing the GPU runtime."""
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/platforms/hcu.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    platform_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HCUPlatform"
    )
    method = next(
        node
        for node in platform_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "supports_fp8"
    )
    namespace = {
        "on_gfx938": lambda: arch == "gfx938",
        "_get_gcn_arch_name": lambda: arch,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    platform = type("HCUPlatform", (), {"supports_fp8": namespace["supports_fp8"]})

    assert platform.supports_fp8() is expected
