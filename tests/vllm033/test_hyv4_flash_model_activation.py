# SPDX-License-Identifier: Apache-2.0
"""CPU opt-in boundary checks; no checkpoint/device execution claimed."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = Path(os.environ.get("HCU_ADAPTER_SOURCE_ROOT") or
               str(Path(importlib.util.find_spec("hcu_adapter").origin).parent.parent))
MODULE = "hcu_vllm_das.hyv4_flash_model_activation"


def run_child(code):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("HCU_ADAPTER_VLLM_REQUIRED_")}
    env.update(HCU_ADAPTER_HOST="none", HIP_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    preamble = ("import sys;sys.path[:0]=" + repr([
        str(ROOT / "adapter_extension/src"), str(ADAPTER),
        sysconfig.get_paths()["purelib"]]) + ";")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", preamble + code],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unselected_discovery_and_activation_take_no_model_ownership():
    run_child("""
import importlib
from types import SimpleNamespace as NS
m = importlib.import_module('hcu_vllm_das.hyv4_flash_model_activation')
s = m.extension_spec()
assert not s.capabilities
s.activate(NS(requested_extensions=set(), requested_capabilities=set()))
s.check_health()
assert m._publication is None
assert not any(name == 'torch' or name.startswith('vllm') or
               name.startswith('hcu_vllm_das.models.hy_v4_flash')
               for name in sys.modules)
""")


@pytest.mark.parametrize("request_first", [True, False])
def test_model_selection_uses_the_same_core_qualification(request_first):
    run_child("""
import importlib
from types import SimpleNamespace as NS
m = importlib.import_module('hcu_vllm_das.hyv4_flash_model_activation')
context = NS(vllm_version='0.32.0', requested_extensions=set(),
             requested_capabilities=set())
request_first = """ + repr(request_first) + """
try:
    if request_first:
        m.request_model_runtime()
        s = m.extension_spec()
        assert s.capabilities == frozenset({'hyv4-flash-model'})
        s.activate(context)
    else:
        s = m.extension_spec()
        assert not s.capabilities
        s.activate(context)
        m.request_model_runtime()
except RuntimeError as e:
    assert 'pinned vLLM 0.33.0' in str(e), e
else:
    raise AssertionError('model selection bypassed Core qualification')
assert m._publication is None
""")


@pytest.mark.parametrize("version", ["0.32.0", "0.34.0", "0.33.0rc1",
                                      "0.33.0.dev1", "0.33.0.post1", "1!0.33.0"])
def test_selected_wrong_core_fails_before_model_import(version):
    run_child("""
import os,importlib
from types import SimpleNamespace as NS
os.environ['HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES'] = 'hyv4-flash-model'
m = importlib.import_module('hcu_vllm_das.hyv4_flash_model_activation')
s = m.extension_spec()
assert s.capabilities == frozenset({'hyv4-flash-model'})
try:
    s.activate(NS(vllm_version=""" + repr(version) + """,
                  requested_extensions=set(),
                  requested_capabilities={'hyv4-flash-model'}))
except RuntimeError as e:
    assert 'pinned vLLM 0.33.0' in str(e), e
else:
    raise AssertionError('unqualified Core version accepted')
assert not any(name.startswith('hcu_vllm_das.models.hy_v4_flash')
               for name in sys.modules)
assert m._publication is None
""")
