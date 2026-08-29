# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static API fingerprint for the audited official vLLM target.

The platform plugin is discovered while vLLM itself is importing, so importing
Worker or Model Runner classes here would create cycles. Instead, inspect the
installed Python sources with AST and compare only contract-facing argument
shapes. Method bodies are audited separately by the upgrade ledger.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeAlias


ArgumentShape: TypeAlias = tuple[
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    tuple[str, ...],
    str | None,
    int,
    tuple[bool, ...],
]


TARGET_API_SHAPES: dict[str, dict[str, ArgumentShape]] = {
    "vllm/plugins/__init__.py": {
        "load_plugins_by_group": ((), ("group",), None, (), None, 0, ()),
        "load_general_plugins": ((), (), None, (), None, 0, ()),
    },
    "vllm/platforms/__init__.py": {
        "resolve_current_platform_cls_qualname": ((), (), None, (), None, 0, ()),
    },
    "vllm/v1/worker/gpu_worker.py": {
        "Worker.__init__": (
            (),
            (
                "self",
                "vllm_config",
                "local_rank",
                "rank",
                "distributed_init_method",
                "is_driver_worker",
            ),
            None,
            (),
            None,
            1,
            (),
        ),
        "Worker.init_device": ((), ("self",), None, (), None, 0, ()),
        "Worker.load_model": (
            (),
            ("self",),
            None,
            ("load_dummy_weights",),
            None,
            0,
            (True,),
        ),
        "Worker.execute_model": (
            (),
            ("self", "scheduler_output"),
            None,
            (),
            None,
            0,
            (),
        ),
    },
    "vllm/v1/worker/gpu_model_runner.py": {
        "GPUModelRunner.__init__": (
            (),
            ("self", "vllm_config", "device"),
            None,
            (),
            None,
            0,
            (),
        ),
        "GPUModelRunner._prepare_inputs": (
            (),
            ("self", "scheduler_output", "num_scheduled_tokens"),
            None,
            (),
            None,
            0,
            (),
        ),
        "GPUModelRunner.execute_model": (
            (),
            ("self", "scheduler_output", "intermediate_tensors"),
            None,
            (),
            None,
            1,
            (),
        ),
        "GPUModelRunner.load_model": (
            (),
            ("self", "load_dummy_weights"),
            None,
            (),
            None,
            1,
            (),
        ),
    },
}


def _argument_shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ArgumentShape:
    arguments = node.args
    return (
        tuple(argument.arg for argument in arguments.posonlyargs),
        tuple(argument.arg for argument in arguments.args),
        None if arguments.vararg is None else arguments.vararg.arg,
        tuple(argument.arg for argument in arguments.kwonlyargs),
        None if arguments.kwarg is None else arguments.kwarg.arg,
        len(arguments.defaults),
        tuple(default is not None for default in arguments.kw_defaults),
    )


def _symbols(tree: ast.Module) -> dict[str, ArgumentShape]:
    result: dict[str, ArgumentShape] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _argument_shape(node)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{member.name}"] = _argument_shape(member)
    return result


def inspect_target_api(vllm_location: str | Path) -> tuple[str, ...]:
    """Return deterministic mismatches against official vLLM v0.28.0."""

    root = Path(vllm_location)
    mismatches: list[str] = []
    for relative_path, expected_symbols in TARGET_API_SHAPES.items():
        path = root / relative_path
        if not path.is_file():
            mismatches.append(f"{relative_path}: missing source")
            continue
        try:
            actual_symbols = _symbols(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, UnicodeError) as error:
            mismatches.append(f"{relative_path}: unreadable source: {error}")
            continue
        for qualified_name, expected_shape in expected_symbols.items():
            actual_shape = actual_symbols.get(qualified_name)
            if actual_shape != expected_shape:
                mismatches.append(
                    f"{relative_path}:{qualified_name}: "
                    f"expected={expected_shape!r}; actual={actual_shape!r}"
                )
    return tuple(mismatches)


