# SPDX-License-Identifier: Apache-2.0
"""The installed portable lane must execute its native-CPU case without opt-in."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def test_default_native_cpu_boundary_passes_real_no_skip_gate(tmp_path):
    environment=dict(os.environ)
    environment.pop("HCU_HYV4_ATTENTION_NATIVE_CPU",None)
    environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",HCU_ADAPTER_HOST="none",
                       HIP_VISIBLE_DEVICES="",ROCR_VISIBLE_DEVICES="",CUDA_VISIBLE_DEVICES="")
    node="tests/vllm033/test_hyv4_flash_active_attention_compile_boundary.py::NativeAttentionBoundaryContracts"
    report=tmp_path/"native-cpu.xml"
    result=subprocess.run([sys.executable,"-B","-m","pytest","-q",node,
                           "--junitxml",str(report)],cwd=ROOT,env=environment,
                          text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    spec=importlib.util.spec_from_file_location("_real_v033_result_gate",ROOT/"tools/run_v033_ci.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate_result(report)
