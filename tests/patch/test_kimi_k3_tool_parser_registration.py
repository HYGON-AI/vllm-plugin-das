# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""K3 XTML tool-parser port: registration contract and wheel-path checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vllm_hcu.patch.platform.core_fix import register_kimi_k3_tool_parser
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

REPO = Path(__file__).resolve().parents[2]


def _fake_module(eager=None, lazy=None):
    class Manager:
        tool_parsers = dict(eager or {})
        lazy_parsers = dict(lazy or {})

        @classmethod
        def register_lazy_module(cls, name, module_path, class_name):
            cls.lazy_parsers[name] = (module_path, class_name)

    module = ModuleType(register_kimi_k3_tool_parser.TARGET_MODULE)
    module.ToolParserManager = Manager
    return module, Manager


def test_registers_kimi_k3_into_the_wheel_manager_once() -> None:
    module, manager = _fake_module()

    assert register_kimi_k3_tool_parser.apply_to_module(module) is True
    assert manager.lazy_parsers["kimi_k3"] == (
        "vllm_hcu.tool_parsers.kimi_k3_tool_parser",
        "KimiK3ToolParser",
    )
    assert register_kimi_k3_tool_parser.apply_to_module(module) is False


def test_rejects_a_conflicting_lazy_owner() -> None:
    module, _ = _fake_module(lazy={"kimi_k3": ("other.module", "OtherParser")})

    with pytest.raises(PatchCompatibilityError, match="already registered"):
        register_kimi_k3_tool_parser.apply_to_module(module)


def test_rejects_a_conflicting_eager_owner() -> None:
    module, _ = _fake_module(eager={"kimi_k3": object()})

    with pytest.raises(PatchCompatibilityError, match="already registered"):
        register_kimi_k3_tool_parser.apply_to_module(module)


def test_rejects_manager_without_lazy_registry() -> None:
    module = ModuleType(register_kimi_k3_tool_parser.TARGET_MODULE)

    class Manager:
        tool_parsers = {}

    module.ToolParserManager = Manager

    with pytest.raises(PatchCompatibilityError, match="lazy_parsers"):
        register_kimi_k3_tool_parser.apply_to_module(module)


def test_ported_parser_module_exposes_the_wheel_contract() -> None:
    """The port keeps the public surface the wheel serve path expects."""

    parser_module = pytest.importorskip(
        "vllm_hcu.tool_parsers.kimi_k3_tool_parser"
    )
    from vllm.tool_parsers.abstract_tool_parser import ToolParser

    parser_class = parser_module.KimiK3ToolParser
    assert issubclass(parser_class, ToolParser)
    for method in (
        "adjust_request",
        "extract_tool_calls",
        "extract_tool_calls_streaming",
    ):
        assert callable(getattr(parser_class, method)), method


def test_service_path_resolves_kimi_k3_through_the_real_manager() -> None:
    code = r"""
from vllm.tool_parsers import ToolParserManager
from vllm_hcu.patch import apply_platform_patches

apply_platform_patches()

tool = ToolParserManager.get_tool_parser("kimi_k3")
assert tool.__module__ == "vllm_hcu.tool_parsers.kimi_k3_tool_parser", tool
assert tool.__name__ == "KimiK3ToolParser", tool
print("registered")
"""
    environment = dict(os.environ)
    environment.pop("VLLM_PLUGINS", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert "registered" in result.stdout
