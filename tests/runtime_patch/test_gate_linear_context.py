# SPDX-License-Identifier: Apache-2.0
"""Real GateLinear/config lifecycle in an isolated live-HCU process."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.fixtures.vllm_source import resolve_target_vllm_root


@pytest.mark.hcu
@pytest.mark.isolated_process
def test_real_gate_constructor_and_forward_after_config_context():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip('requires a live HCU/ROCm device')
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, VLLM_PLUGINS='__disabled__')
    env['PYTHONPATH'] = os.pathsep.join((str(repo), str(resolve_target_vllm_root())))
    result = subprocess.run(
        [sys.executable, str(repo / 'tools/validate_deepseek_v4_gate.py'),
         '--construction-only'], env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.hcu
@pytest.mark.isolated_process
@pytest.mark.parametrize('hidden_size,experts,scale', [(4096, 256, 1.5), (7168, 384, 2.5)])
def test_real_gate_routing_eager_and_graph(hidden_size, experts, scale, tmp_path):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip('requires a live HCU/ROCm device')
    import json

    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, VLLM_PLUGINS='__disabled__')
    env['PYTHONPATH'] = os.pathsep.join((str(repo), str(resolve_target_vllm_root())))
    output = tmp_path / 'routing.json'
    result = subprocess.run(
        [sys.executable, str(repo / 'tools/validate_deepseek_v4_gate.py'),
         '--hidden-size', str(hidden_size), '--experts', str(experts),
         '--routed-scaling-factor', str(scale), '--tokens', '1', '17',
         '--seeds', '42', '--output', str(output)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())['comparisons']
    assert len(rows) == 8  # Two layouts, two token counts, eager and graph.
    assert {r['mode'] for r in rows} == {'eager', 'graph'}
    assert all(r['eager_graph_logits_exact'] for r in rows)
