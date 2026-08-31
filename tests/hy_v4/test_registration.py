# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
    environment["VLLM_PLUGINS"] = "__disabled__"
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
