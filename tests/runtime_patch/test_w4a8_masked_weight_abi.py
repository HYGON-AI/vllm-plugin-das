"""Verify N32 storage is preserved at the masked HIPC rank-3 boundary."""
import ast
from pathlib import Path

import pytest
import torch


def load_adapter():
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_w4a8_masked_kernel_weight")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("shape", [(2, 128, 64), (3, 64, 128)])
def test_n32_adapter_preserves_storage_and_bytes(shape):
    adapt = load_adapter()
    packed = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64).to(torch.int8).view(shape)
    e, n, k_half = shape
    n32 = packed.view(e, k_half // 32, n // 32, 4, 32, 8)
    actual = adapt(n32)
    assert actual.shape == packed.shape
    assert actual.data_ptr() == packed.data_ptr()
    torch.testing.assert_close(actual, packed)
    assert adapt(packed) is packed


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError):
        load_adapter()(torch.empty(1, 2, 3, 4))
