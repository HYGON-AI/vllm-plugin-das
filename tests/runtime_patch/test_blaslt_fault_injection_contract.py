# SPDX-License-Identifier: Apache-2.0
"""Exercise the real JIT source and injection path without requiring HIP."""

import ast
import functools
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.utils.cpp_extension as cpp_extension

from tests.fixtures.blaslt_faults import (
    BLASLT_FAILURE_IDS,
    BLASLT_FAILURE_SITES,
    BLASLT_FAILURE_STATUSES,
    failing_blaslt_loader,
)


@pytest.mark.parametrize(
    "api,occurrence,expected_calls", BLASLT_FAILURE_SITES, ids=BLASLT_FAILURE_IDS,
)
@pytest.mark.parametrize("status", BLASLT_FAILURE_STATUSES)
def test_injected_jit_reaches_compiler(monkeypatch, api, occurrence, expected_calls, status):
    path = Path(__file__).parents[2] / "vllm_hcu/utils/bf16_fp32_gemm.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_jit_hipblaslt_bf16_fp32"
    )
    namespace = {"functools": functools, "torch": torch, "Any": Any}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    jit = namespace[function.name]
    calls = []
    sentinel = object()

    def compiler(**kwargs):
        calls.append(kwargs)
        return sentinel

    # First capture the unchanged production source at the compilation boundary.
    monkeypatch.setattr(cpp_extension, "load_inline", compiler)
    assert jit() is sentinel
    original = calls.pop()
    jit.cache_clear()
    monkeypatch.setattr(
        cpp_extension, "load_inline",
        failing_blaslt_loader(compiler, api, status, occurrence, expected_calls),
    )
    assert jit() is sentinel
    assert len(calls) == 1
    injected = calls[0]
    source = original["cuda_sources"]
    call = api + "("
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(call, start + 1)
    replacement = f"([](auto... args) {{ return {status}; }})("
    assert injected["cuda_sources"] == (
        source[:start] + replacement + source[start + len(call):]
    )
    assert injected["cuda_sources"].count(call) == expected_calls - 1
    assert injected["name"] == original["name"] + f"_{api}_{occurrence}_{status}"
