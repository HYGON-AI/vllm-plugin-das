# SPDX-License-Identifier: Apache-2.0
"""Check DTK graph defaults in a fresh process before HIP initialization."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("configured", [None, "", "0", "1"])
def test_plugin_configures_graph_command_cache_before_accelerator_import(configured):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY)
    env.pop("HIP_GRAPH_USE_CMD_CACHE", None)
    env.pop("HIP_LAUNCH_BLOCKING", None)
    if configured is not None:
        env["HIP_GRAPH_USE_CMD_CACHE"] = configured
    expected = configured or "0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys\n"
            "import vllm_hcu\n"
            f"assert os.environ.get('HIP_GRAPH_USE_CMD_CACHE') == {expected!r}\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'flash_attn' not in sys.modules\n"
            "assert 'HIP_LAUNCH_BLOCKING' not in os.environ\n",
        ],
        cwd=REPOSITORY,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
