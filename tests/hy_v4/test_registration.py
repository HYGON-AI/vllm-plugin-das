# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
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
from vllm_hcu.patch.runtime_state import PATCH_REGISTRY


REPO = Path(__file__).resolve().parents[2]


def _manager_target(
    adapter,
    manager_name: str,
    eager_name: str,
    *,
    eager=None,
    lazy=None,
):
    class Manager:
        lazy_parsers = dict(lazy or {})

        @classmethod
        def register_lazy_module(cls, name, module_path, class_name):
            cls.lazy_parsers[name] = (module_path, class_name)

    setattr(Manager, eager_name, dict(eager or {}))
    target = ModuleType(adapter.TARGET_MODULE)
    setattr(target, manager_name, Manager)
    return target, Manager


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
    ("adapter", "manager_name", "eager_name"),
    (
        (
            register_hy_v4_reasoning_parser,
            "ReasoningParserManager",
            "reasoning_parsers",
        ),
        (register_hy_v4_tool_parser, "ToolParserManager", "tool_parsers"),
    ),
)
def test_parser_registration_accepts_only_the_identical_owner(
    adapter, manager_name, eager_name
) -> None:
    target, Manager = _manager_target(adapter, manager_name, eager_name)

    assert adapter.apply_to_module(target) is True
    assert adapter.apply_to_module(target) is False
    assert Manager.lazy_parsers["hy_v4"] == (
        adapter._PARSER_MODULE,
        adapter._PARSER_CLASS,
    )

    foreign_target, _ = _manager_target(
        adapter,
        manager_name,
        eager_name,
        lazy={"hy_v4": ("foreign.parser", "ForeignParser")},
    )
    with pytest.raises(PatchCompatibilityError, match="already registered as"):
        adapter.apply_to_module(foreign_target)


@pytest.mark.parametrize(
    ("adapter", "manager_name", "eager_name"),
    (
        (
            register_hy_v4_reasoning_parser,
            "ReasoningParserManager",
            "reasoning_parsers",
        ),
        (register_hy_v4_tool_parser, "ToolParserManager", "tool_parsers"),
    ),
)
def test_parser_registration_rejects_foreign_eager_owner(
    adapter, manager_name, eager_name
) -> None:
    class ForeignParser:
        pass

    target, _ = _manager_target(
        adapter,
        manager_name,
        eager_name,
        eager={"hy_v4": ForeignParser},
    )

    with pytest.raises(PatchCompatibilityError, match="already registered as"):
        adapter.apply_to_module(target)


@pytest.mark.parametrize(
    ("adapter", "manager_name", "eager_name"),
    (
        (
            register_hy_v4_reasoning_parser,
            "ReasoningParserManager",
            "reasoning_parsers",
        ),
        (register_hy_v4_tool_parser, "ToolParserManager", "tool_parsers"),
    ),
)
def test_parser_registration_accepts_identical_loaded_owner(
    adapter, manager_name, eager_name
) -> None:
    parser_module = importlib.import_module(adapter._PARSER_MODULE)
    parser_class = getattr(parser_module, adapter._PARSER_CLASS)
    target, Manager = _manager_target(
        adapter,
        manager_name,
        eager_name,
        eager={"hy_v4": parser_class},
        lazy={"hy_v4": (adapter._PARSER_MODULE, adapter._PARSER_CLASS)},
    )

    assert adapter.apply_to_module(target) is True
    assert adapter.apply_to_module(target) is False
    assert getattr(Manager, eager_name)["hy_v4"] is parser_class


@pytest.mark.parametrize(
    ("adapter", "manager_name", "eager_name", "conflict_registry"),
    (
        (
            register_hy_v4_reasoning_parser,
            "ReasoningParserManager",
            "reasoning_parsers",
            "eager",
        ),
        (
            register_hy_v4_reasoning_parser,
            "ReasoningParserManager",
            "reasoning_parsers",
            "lazy",
        ),
        (
            register_hy_v4_tool_parser,
            "ToolParserManager",
            "tool_parsers",
            "eager",
        ),
        (
            register_hy_v4_tool_parser,
            "ToolParserManager",
            "tool_parsers",
            "lazy",
        ),
    ),
)
def test_parser_registration_revalidates_ownership_after_initial_registration(
    adapter, manager_name, eager_name, conflict_registry
) -> None:
    target, Manager = _manager_target(adapter, manager_name, eager_name)
    PATCH_REGISTRY.reset_for_tests()
    try:
        assert adapter.apply(target) is True

        if conflict_registry == "eager":
            class ForeignParser:
                pass

            getattr(Manager, eager_name)["hy_v4"] = ForeignParser
        else:
            Manager.lazy_parsers["hy_v4"] = ("foreign.parser", "ForeignParser")

        with pytest.raises(PatchCompatibilityError, match="already registered as"):
            adapter.apply(target)
    finally:
        PATCH_REGISTRY.reset_for_tests()
