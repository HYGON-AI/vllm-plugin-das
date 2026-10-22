# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""IPC module imports must not require an active Triton device driver."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("has_constexpr", [False, True])
def test_ipc_import_supports_placeholder_and_real_language_modules(
    monkeypatch: pytest.MonkeyPatch,
    has_constexpr: bool,
) -> None:
    constructions = []
    kernels = []

    class Constexpr:
        def __init__(self, value):
            constructions.append(value)
            self.value = value

    def jit(function):
        kernels.append(function.__name__)
        return function

    shim = ModuleType("vllm.triton_utils")
    shim.HAS_TRITON = has_constexpr
    shim.tl = SimpleNamespace(constexpr=Constexpr if has_constexpr else None)
    shim.triton = SimpleNamespace(jit=jit)
    monkeypatch.setitem(sys.modules, shim.__name__, shim)
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/distributed/device_communicators/ipc_all_gather.py"
    )
    spec = importlib.util.spec_from_file_location("_ipc_import_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert kernels == ["_ipc_all_gather_kernel", "_zero_i64_kernel"]
    assert callable(module.IpcAllGather)
    if has_constexpr:
        assert constructions == [4]
        assert isinstance(module._NT, Constexpr)
        assert module._NT.value == 4
    else:
        assert constructions == []
        assert module._NT == 4
