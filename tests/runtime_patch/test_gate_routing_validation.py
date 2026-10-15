# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the HCU validator's tied top-k checks."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def validator():
    source = Path(__file__).resolve().parents[2] / "tools/validate_deepseek_v4_gate.py"
    spec = importlib.util.spec_from_file_location("gate_routing_validator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_topk_routing


def routing_case(ids):
    logits = torch.full((1, 256), -3.0)
    logits[0, :7] = torch.tensor([3., 2., 1., 0., -1., -1., -1.])
    ids = torch.tensor([ids])
    selected = F.softplus(logits).sqrt().gather(1, ids)
    weights = selected / selected.sum(-1, keepdim=True) * 1.5
    return logits, ids, weights


@pytest.mark.parametrize("ids", [
    [0, 1, 2, 3, 4, 5],
    [6, 3, 2, 1, 0, 4],  # Alternate tie IDs and arbitrary output order.
])
def test_accepts_complete_topk_with_ties(validator, ids):
    validator(*routing_case(ids), 6, 1.5)


@pytest.mark.parametrize("ids", [
    [0, 1, 2, 4, 5, 6],  # Omit a higher score; replace with a cutoff tie.
    [0, 1, 2, 3, 4, 7],  # Select below cutoff.
    [0, 1, 2, 3, 4, 4],  # Correct score multiset but duplicate expert IDs.
])
def test_rejects_invalid_topk_even_with_self_consistent_weights(validator, ids):
    with pytest.raises(AssertionError):
        validator(*routing_case(ids), 6, 1.5)


def test_rejects_wrong_weights_for_valid_topk(validator):
    logits, ids, weights = routing_case([0, 1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        validator(logits, ids, torch.zeros_like(weights), 6, 1.5)
