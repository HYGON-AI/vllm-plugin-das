# SPDX-License-Identifier: Apache-2.0
"""Check Engram storage configuration without importing the accelerator runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def offload():
    path = Path(__file__).resolve().parents[2] / (
        'adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/engram_ops.py'
    )
    tree = ast.parse(path.read_bytes())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'get_engram_cpu_offload')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    return scope[function.name]


def test_offload_is_opt_in_and_preserves_the_core_configuration(offload):
    key = 'hyv4_flash_engram_cpu_offload'
    assert not offload(None)
    assert not offload(None, additional_config={key: False})
    assert offload(None, additional_config={key: True})
    assert offload(SimpleNamespace(cpu_offload=True))


def test_offload_rejects_invalid_and_conflicting_configuration(offload):
    key = 'hyv4_flash_engram_cpu_offload'
    with pytest.raises(ValueError, match='boolean'):
        offload(None, additional_config={key: 1})
    with pytest.raises(ValueError, match='conflicts'):
        offload(SimpleNamespace(cpu_offload=False), additional_config={key: True})
    with pytest.raises(ValueError, match='TP-local'):
        offload(SimpleNamespace(cpu_offload=True, embedding_across_dp=True))
