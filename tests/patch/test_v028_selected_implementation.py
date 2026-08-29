# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _class(path: str, name: str) -> ast.ClassDef:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        member.name: member
        for member in node.body
        if isinstance(member, ast.FunctionDef)
    }


def test_platform_inherits_target_rocm_and_has_no_empty_block_hook() -> None:
    platform = _class("vllm_hcu/platforms/hcu.py", "HCUPlatform")

    assert [ast.unparse(base) for base in platform.bases] == ["RocmPlatform"]
    methods = _methods(platform)
    assert "update_block_size_for_backend" not in methods
    assert {
        "get_valid_backends",
        "apply_config_platform_defaults",
        "check_and_update_config",
        "supports_fp8",
    } <= methods.keys()


def test_worker_keeps_target_runner_and_lifecycle_selection() -> None:
    worker_path = REPO_ROOT / "vllm_hcu/v1/worker.py"
    worker_tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HcuGPUWorker"
    )

    assert [ast.unparse(base) for base in worker.bases] == ["Worker"]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_create_model_runner"
        for node in worker_tree.body
    )
    methods = _methods(worker)
    assert set(methods) == {"__init__", "load_model", "init_device"}
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
        and call.func.attr == "init_device"
        for call in ast.walk(methods["init_device"])
        if isinstance(call, ast.Call)
    )
