# SPDX-License-Identifier: Apache-2.0
"""GPU parity with LightOp, including invalid DeepGEMM workspace rows."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("width", [512, 1024, 1536, 2048])
@pytest.mark.parametrize("stride_factor", [1, 2])
def test_workspace_offsets_do_not_overflow_int32(width, stride_factor):
    path = (Path(__file__).resolve().parents[2] / "vllm_hcu/model_executor/"
            "layers/quantization/slimquant_w4a8_activation.py")
    kernel = next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_masked_clamp_quant")
    row_expr = next(node.value for node in kernel.body
                    if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "row")
    # Evaluate the actual address expressions on tiny CPU tensors at both
    # input (2 * D) and output (D) signed-INT32 overflow boundaries.
    first_overflow_row = ((1 << 31) + stride_factor * width - 1) // (
        stride_factor * width)
    tile = 8
    block = first_overflow_row // tile
    tl = SimpleNamespace(
        program_id=lambda axis: torch.tensor(block, dtype=torch.int32),
        arange=lambda start, end: torch.arange(start, end, dtype=torch.int32),
        int64=torch.int64,
    )
    namespace = {"tl": tl, "ROW_TILE": tile, "D": width,
                 "col": torch.tensor([0, width - 1], dtype=torch.int32),
                 "X": 0, "Y": 0}

    def evaluate(expr):
        return eval(compile(ast.Expression(expr), str(path), "eval"), namespace)

    namespace["row"] = evaluate(row_expr)
    expected_rows = torch.arange(block * tile, (block + 1) * tile,
                                 dtype=torch.int64)[:, None]
    expected_col = namespace["col"].to(torch.int64)[None, :]
    offsets = []
    for node in ast.walk(kernel):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("load", "store")):
            expr = node.args[0]
            names = {item.id for item in ast.walk(expr)
                     if isinstance(item, ast.Name)}
            if "X" in names or "Y" in names:
                offsets.append((expr, names))
    assert len(offsets) == 3  # gate, up, output
    for expr, names in offsets:
        expected = expected_rows * (2 * width if "X" in names else width)
        if "X" in names and " + D + " in ast.unparse(expr):
            expected = expected + width
        actual = evaluate(expr)
        assert actual.dtype == torch.int64
        torch.testing.assert_close(actual, expected + expected_col, rtol=0, atol=0)


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
