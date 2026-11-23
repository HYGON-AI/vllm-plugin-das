import os
from pathlib import Path
import subprocess


def test_service_log_deduplication_preserves_diagnostics_and_exit_status(tmp_path):
    wrapper = Path(__file__).resolve().parents[2] / "scripts/serve_hy4_megamoe.sh"
    vllm = tmp_path / "vllm"
    vllm.write_text('#!/bin/sh\n[ "$1" = serve ] && [ "$2" = "model path" ] || exit 2\n'
                    'cat >&2\nexit 17\n')
    vllm.chmod(0o755)
    warning = ('Warning: (!g_use_interrupt_wait || isIPC()) && '
               '"Use of non-host signal in host signal wait API." in virtual '
               'hsa_signal_value_t rocr::core::BusyWaitSignal::WaitRelaxed(condition), '
               '/build/default_signal.cpp:56\n')
    first = "(Worker_PCP0_EP0 pid=10) [rank 0] SymmBuffer ipc: 483 MB\n"
    duplicate = "[rank 1] SymmBuffer ipc: 483 MB\n"
    changed = "[rank 1] SymmBuffer ipc: 600 MB\n"
    other = warning.replace("BusyWaitSignal::WaitRelaxed", "OtherSignal::WaitRelaxed")
    log = warning + first + duplicate + warning + changed + other * 2
    result = subprocess.run(
        ["bash", str(wrapper), "model path"], input=log, text=True, capture_output=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )
    assert result.returncode == 17
    assert result.stdout == warning + first + changed + other * 2
