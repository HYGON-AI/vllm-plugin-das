"""CPU-only proof for the runtime FP4 packer lowering patch."""

from __future__ import annotations

import inspect
import math
import types

import torch

from vllm_hcu.patch.worker.core_fix import patch_fp4_packer_lowering as patch


def _reference_code(x: torch.Tensor) -> torch.Tensor:
    # Pure-torch model of cvt.rn.satfinite.e2m1x2.f32.  The sign is
    # unconditional, including negative zero; ties use the even code.
    return patch.e2m1_code_arithmetic_reference(x)


def _nextafter(values: list[float]) -> torch.Tensor:
    result = []
    for value in values:
        result.extend((math.nextafter(value, -math.inf), value, math.nextafter(value, math.inf)))
        result.extend((math.nextafter(-value, -math.inf), -value, math.nextafter(-value, math.inf)))
    return torch.tensor(result, dtype=torch.float32)


def test_e2m1_domain_grid_ties_saturation_and_denormals() -> None:
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    values = torch.cat(
        (
            torch.tensor(
                [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5,
                 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0,
                 -0.1, -1.0e-8], dtype=torch.float32
            ),
            _nextafter(ties),
            torch.tensor(
                [float("inf"), float("-inf"), float("nan"), 7.0, -7.0,
                 torch.finfo(torch.float32).tiny,
                 torch.finfo(torch.float32).tiny * 0.5], dtype=torch.float32
            ),
        )
    )
    actual = _reference_code(values)
    # Exact ties map to even E2M1 magnitudes: 0, 1, 2, 4, 4, 6, 6.
    tie_expected = torch.tensor([0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], dtype=torch.uint8)
    tie_values = torch.tensor([v for v in ties for v in (v, -v)], dtype=torch.float32)
    assert torch.equal(_reference_code(tie_values) & 7, tie_expected)
    assert int(_reference_code(torch.tensor([-0.1]))[0]) == 0x8
    assert int(_reference_code(torch.tensor([-0.0]))[0]) == 0x8
    assert int(_reference_code(torch.tensor([-1.0e-8]))[0]) == 0x8
    assert int(_reference_code(torch.tensor([7.0]))[0]) == 7
    assert int(_reference_code(torch.tensor([-7.0]))[0]) == 15
    assert int(_reference_code(torch.tensor([float("inf")]))[0]) == 7
    assert int(_reference_code(torch.tensor([float("-inf")]))[0]) == 15
    assert int(_reference_code(torch.tensor([float("nan")]))[0]) == 0
    assert actual.shape == values.shape
    assert not bool(torch.isnan(actual.float()).any())


def test_payload_matches_pure_torch_inline_model_and_nibble_order() -> None:
    torch.manual_seed(20261018)
    boundaries = [v for v in (0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0) for v in (v, -v)]
    values = torch.cat(
        (torch.randn(4096, dtype=torch.float32) * 3.0,
         torch.tensor(boundaries * 8, dtype=torch.float32))
    )
    lo = values[0::2]
    hi = values[1::2]
    expected = _reference_code(lo) | (_reference_code(hi) << 4)
    actual = patch.e2m1_payload_reference(lo, hi)
    assert torch.equal(actual, expected)
    assert torch.equal(actual & 0x0F, _reference_code(lo))
    assert torch.equal(actual >> 4, _reference_code(hi))


def test_fail_closed_and_flag_off_identity(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FP4_PACKER_LOWERING", "0")
    module = types.ModuleType(patch.TARGET_MODULE)
    original = object()
    module._fp32x2_to_fp4x2 = original
    assert patch.apply_to_module(module) is False
    assert module._fp32x2_to_fp4x2 is original
    assert module._vllm_hcu_fp4_packer_lowering_status == "OFF"

    changed = types.ModuleType(patch.TARGET_MODULE)
    changed._fp32x2_to_fp4x2 = lambda x, y: x
    monkeypatch.setenv("VLLM_HCU_USE_FP4_PACKER_LOWERING", "1")
    assert patch.apply_to_module(changed) is False
    assert changed._vllm_hcu_fp4_packer_lowering_status == "UPSTREAM_CHANGED"
    assert changed._fp32x2_to_fp4x2.__name__ == "<lambda>"


def test_patch_module_has_no_device_initialization() -> None:
    source = inspect.getsource(patch)
    assert "torch.cuda" not in source
    assert "torch.device" not in source
    triton_source = inspect.getsource(patch._fp32x2_to_fp4x2_arithmetic.fn)
    assert "inline_asm_elementwise" not in triton_source
    code_source = inspect.getsource(patch._e2m1_code_arithmetic.fn)
    assert "0.25" in code_source and "5.0" in code_source
    assert "bitcast=True" in code_source
    assert patch.TARGETS == (
        "vllm.models.deepseek_v41.common.ops.indexer_k_store._fp32x2_to_fp4x2",
    )
    assert patch._fp32x2_to_fp4x2_arithmetic is not None
