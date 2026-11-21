"""Installed CI accepts the same PEP 440 release builds as activation."""
import importlib.util
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    '_hcu_v033_ci_release', Path(__file__).resolve().parents[2] / 'tools/run_v033_ci.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.mark.parametrize('version', [
    '0.33.0', '0.33.0+das.da8b48faa.dtk2610', '0.33.0rc1',
    '0.33.0.dev7', '0.33.0.post1', '0!0.33.0',
])
def test_compatible_release_builds(version):
    assert runner._vllm_release(version)[:2] == (0, 33)


@pytest.mark.parametrize('version', ['0.32.9', '0.34.0', '0.330.0'])
def test_other_release_lines_do_not_match(version):
    assert runner._vllm_release(version)[:2] != (0, 33)


@pytest.mark.parametrize('version', ['1!0.33.0', 'not-a-version'])
def test_epoch_and_invalid_versions_refuse(version):
    with pytest.raises(ValueError):
        runner._vllm_release(version)
