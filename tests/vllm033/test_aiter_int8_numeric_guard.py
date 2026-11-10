"""A shared zero-output defect must not satisfy the nondegenerate INT8 gate."""
from pathlib import Path
import runpy

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_full_ct_gate_refuses_shared_zero_output(dtype):
    helper = runpy.run_path(str(Path(__file__).with_name('test_aiter_int8_ct_graph_hcu.py')))['_metrics']
    reference = torch.zeros((2, 8), dtype=dtype)
    with pytest.raises(AssertionError):
        helper(reference.clone(), reference)
