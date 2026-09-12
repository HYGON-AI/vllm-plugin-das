# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vllm_hcu.patch.platform.core_fix import (
    register_hy_v4_reasoning_parser,
    register_hy_v4_tool_parser,
)
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError


REPO = Path(__file__).resolve().parents[2]


def test_platform_plugin_registers_hy_v4_parsers_with_vllm_managers() -> None:
    code = r"""
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm_hcu.patch import apply_platform_patches

apply_platform_patches()

reasoning = ReasoningParserManager.get_reasoning_parser("hy_v4")
tool = ToolParserManager.get_tool_parser("hy_v4")
assert reasoning.__module__ == "vllm_hcu.reasoning.hy_v4_reasoning_parser"
assert reasoning.__name__ == "HYV4ReasoningParser"
assert tool.__module__ == "vllm_hcu.tool_parsers.hy_v4_tool_parser"
assert tool.__name__ == "HYV4ToolParser"
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
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("registered")


@pytest.mark.parametrize(
    ("adapter", "manager_name"),
    (
        (register_hy_v4_reasoning_parser, "ReasoningParserManager"),
        (register_hy_v4_tool_parser, "ToolParserManager"),
    ),
)
def test_parser_registration_accepts_only_the_identical_owner(
    adapter, manager_name
) -> None:
    class Manager:
        lazy_parsers: dict[str, tuple[str, str]] = {}

        @classmethod
        def register_lazy_module(cls, name, module_path, class_name):
            cls.lazy_parsers[name] = (module_path, class_name)

    target = ModuleType(adapter.TARGET_MODULE)
    setattr(target, manager_name, Manager)

    assert adapter.apply_to_module(target) is True
    assert adapter.apply_to_module(target) is False
    assert Manager.lazy_parsers["hy_v4"] == (
        adapter._PARSER_MODULE,
        adapter._PARSER_CLASS,
    )

    class ForeignManager:
        lazy_parsers = {"hy_v4": ("foreign.parser", "ForeignParser")}

        @classmethod
        def register_lazy_module(cls, name, module_path, class_name):
            cls.lazy_parsers[name] = (module_path, class_name)

    foreign_target = ModuleType(adapter.TARGET_MODULE)
    setattr(foreign_target, manager_name, ForeignManager)
    with pytest.raises(PatchCompatibilityError, match="already registered as"):
        adapter.apply_to_module(foreign_target)
