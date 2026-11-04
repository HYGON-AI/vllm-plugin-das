# SPDX-License-Identifier: Apache-2.0
"""Capture CLI preparation without starting a GPU service."""
from pathlib import Path
import subprocess
import sys

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools/capture_static_eplb.py"


@pytest.mark.parametrize("delayed", [False, True])
def test_prepare_capture_gate(tmp_path, delayed):
    run = tmp_path / "run.sh"
    continuation = " " + chr(92) + "\n"
    run.write_text(
        "vllm serve /model --enable-eplb --port 10018" + continuation
        + "  --eplb-config '{}'" + continuation + "  2>&1 | tee old.log\n"
    )
    bench = tmp_path / "bench.sh"
    bench.write_text("true\n")
    output = tmp_path / "plan with spaces.json"
    command = [sys.executable, str(TOOL), "--run-script", str(run),
               "--bench-script", str(bench), "--output", str(output), "--prepare-only"]
    if delayed:
        command.append("--start-after-warmup")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    generated = Path(str(output) + ".capture.sh").read_text()
    assert not output.exists()
    assert not Path(str(output) + ".capture-start").exists()
    assert "expert_map_record_path" in generated
    if delayed:
        assert "export VLLM_HCU_EPLB_CAPTURE_START_FILE=" in generated
        assert str(output) + ".capture-start" in generated
    else:
        assert "unset VLLM_HCU_EPLB_CAPTURE_START_FILE" in generated


def test_reject_stale_start_signal(tmp_path):
    output = tmp_path / "plan.json"
    Path(str(output) + ".capture-start").touch()
    result = subprocess.run(
        [sys.executable, str(TOOL), "--run-script", "unused", "--bench-script", "unused",
         "--output", str(output), "--start-after-warmup", "--prepare-only"],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "start signal already exists" in result.stderr
