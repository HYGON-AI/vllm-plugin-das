from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PLUGIN_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_vllm_das.environment import register_compile_factors, rms_norm_enabled


def test_compile_factor_registration_preserves_other_owners():
    other = lambda: "untouched"
    envs = SimpleNamespace(
        environment_variables={"OTHER": other}, compile_factors=lambda: {}
    )
    register_compile_factors(envs)
    register_compile_factors(envs)
    assert envs.environment_variables["OTHER"] is other
    assert (
        envs.environment_variables["VLLM_HCU_DAS_RMS_NORM_ENABLED"]
        is rms_norm_enabled
    )
    envs.environment_variables["VLLM_HCU_DAS_RMS_NORM_ENABLED"] = other
    with pytest.raises(RuntimeError, match="another owner"):
        register_compile_factors(envs)


@pytest.mark.parametrize(
    "envs", [SimpleNamespace(), SimpleNamespace(environment_variables={})]
)
def test_missing_compile_registry_fails_explicitly(envs):
    with pytest.raises(RuntimeError, match="compile-factor registry"):
        register_compile_factors(envs)


def test_actual_vllm_aot_hash_tracks_effective_policy():
    # Exercise the target's actual AOT cache-key function, not a copy of it.
    code = f"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, {os.fspath(PLUGIN_SOURCE_ROOT)!r})
from hcu_vllm_das.environment import register_compile_factors
import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
register_compile_factors(envs)
config = SimpleNamespace(compute_hash=lambda: 'unchanged-model-config')
hashes = {{}}
for master, child in [('0','0'), ('0','1'), ('1','0'), ('1','1'), ('true','yes')]:
    os.environ['VLLM_HCU_USE_CUSTOM_OPS'] = master
    os.environ['VLLM_HCU_USE_CUSTOM_RMS_NORM'] = child
    hashes[(master, child)] = aot_compile_hash_factors(config)
assert hashes[('0','0')] == hashes[('0','1')] == hashes[('1','0')]
assert hashes[('1','1')] == hashes[('true','yes')]
assert hashes[('0','0')] != hashes[('1','1')]
assert hashes[('0','0')][1] == hashes[('1','1')][1]
assert envs.compile_factors()['VLLM_HCU_DAS_RMS_NORM_ENABLED'] is True
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
