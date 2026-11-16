# SPDX-License-Identifier: Apache-2.0
"""Run the production Triton prefill kernels on CPU through its interpreter."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [64, 192])
def test_long_prefill_respects_each_kv_groups_selection(groups, head_dim):
    # A subprocess enables the interpreter before any Triton decorators run.
    script = r"""
import importlib.util
import sys
import torch

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("dsa", sys.argv[1])
dsa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dsa
spec.loader.exec_module(dsa)
dsa._shared_memory_limit = lambda: 0
groups, dim = map(int, sys.argv[2:])
total_q, heads_per_group, slots = 2, 2, 513
q = torch.zeros(total_q, groups * heads_per_group, dim, dtype=torch.float16)
k = torch.zeros(4112, groups, dim, dtype=q.dtype)
v = torch.zeros_like(k)
logical = torch.empty(groups * total_q, slots, dtype=torch.int32)
for g in range(groups):
    # Two valid top-512 choices; group zero excludes the only nonzero region.
    shift = g % 2
    choice = torch.cat((torch.arange(shift, 512 + shift), torch.tensor([513])))
    logical[g * total_q:(g + 1) * total_q] = choice
    v[4096:4104, g] = g + 1
packed = logical | (8 << 24)
counts = torch.full((groups * total_q,), slots, dtype=torch.int32)
positions = torch.tensor([4104, 4105], dtype=torch.int32).repeat(groups)
# Exercise group-major positions too, with a different tail for later groups.
positions[total_q::total_q] = 4103
out, lse = dsa.sparse_attention_prefill(
    q, k, v, packed, counts, num_kv_groups=groups,
    logical_regions=logical, q_positions=positions,
    request_indices=torch.zeros(groups * total_q, dtype=torch.int32),
    region_size=8, block_q=32, block_regions=16,
)
expected = torch.zeros_like(out)
expected_lse = torch.empty_like(lse)
for g in range(groups):
    for t in range(total_q):
        pos = int(positions[g * total_q + t])
        selected = logical[g * total_q + t].long()
        tokens = (selected[:, None] * 8 + torch.arange(8)).flatten()
        tokens = tokens[tokens <= pos]
        value = v[tokens, g].float().mean(0)
        expected[t, g * heads_per_group:(g + 1) * heads_per_group] = value
        expected_lse[t, g * heads_per_group:(g + 1) * heads_per_group] = torch.tensor(float(tokens.numel())).log()
torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.004)
torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=1e-5)
"""
    source = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/step4_dsa_kernels.py"
    )
    env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(groups), str(head_dim)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
