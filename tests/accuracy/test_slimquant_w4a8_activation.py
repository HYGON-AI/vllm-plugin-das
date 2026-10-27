# SPDX-License-Identifier: Apache-2.0
"""GPU parity with LightOp, including invalid DeepGEMM workspace rows."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("width", [512, 1024, 1536, 2048])
@pytest.mark.parametrize("layout", ["empty", "mixed", "all"])
def test_masked_clamp_quant_matches_lightop(width, layout):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("Requires HCU and LightOp")
    lightop = pytest.importorskip("lightop")
    path = (Path(__file__).resolve().parents[2] / "vllm_hcu/model_executor/"
            "layers/quantization/slimquant_w4a8_activation.py")
    spec = importlib.util.spec_from_file_location("w4a8_activation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    x = torch.randn(257, 2 * width, device="cuda", dtype=torch.bfloat16) * 10
    x[0].zero_()
    # Exercise upper-only gate clamping, symmetric up clamping, and rounding.
    x[1, :8] = torch.tensor(
        [-80, -7, -1, 0, 1, 6.96875, 7, 80], device="cuda")
    ids = torch.full((257,), -1, device="cuda", dtype=torch.int32)
    if layout == "mixed":
        ids[::3] = 0
        ids[1] = 1
    elif layout == "all":
        ids.fill_(0)
    x[ids < 0] = float("nan")
    output = torch.full((257, width), -99, device="cuda", dtype=torch.int8)
    reference, reference_scale = lightop.fuse_silu_mul_clamp_quant(x, limit=7.0)
    actual, scales = module.masked_silu_mul_clamp_quant(
        x, limit=7.0, output=output, expert_ids=ids)
    torch.cuda.synchronize()
    valid = ids >= 0
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual[valid], reference[valid], rtol=0, atol=0)
    torch.testing.assert_close(scales[valid], reference_scale[valid], rtol=0, atol=0)
    assert (actual[~valid] == -99).all()
