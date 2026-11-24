# SPDX-License-Identifier: Apache-2.0
"""Run the complete owner lifecycle cohort in its own HIP interpreter.

The original suite's 32MiB allocator limit belongs to that child only. The
ordinary device job retains its allocator state for all other provider tests.
"""
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

pytestmark = pytest.mark.hcu


def test_complete_hisparse_owner_gpu_lifecycle(tmp_path):
    assert (os.environ.get("VLLM033_HCU_TEST") == "1"
            or os.environ.get("HCU_OWNER_GPU_LIFECYCLE") == "1"), (
                "requires the authorized device tier or explicit physical GPU2 selection")
    suite = Path(__file__).with_name("_hisparse_owner_gpu_lifecycle.py")
    report = tmp_path / "owner-lifecycle.xml"
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTHONPATH", None)
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-s", "--strict-markers",
         "-o", "xfail_strict=true", "-o", f"cache_dir={tmp_path / 'cache'}",
         "--junitxml", str(report), str(suite)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    print(run.stdout)
    print(run.stderr, file=sys.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert cases and not any(case.find(tag) is not None for case in cases
                             for tag in ("failure", "error", "skipped"))
