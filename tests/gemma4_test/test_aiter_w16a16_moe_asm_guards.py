# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_aiter_ops_asm_branch_is_w16a16_only() -> None:
    source = (
        REPO_ROOT / "vllm_hcu/patches/vllm___aiter_ops.patch.py"
    ).read_text()

    assert "use_w16a16_asm = (" in source
    assert "quant_type == QuantType.No" in source
    assert "w1_scale is None" in source
    assert "w2_scale is None" in source
    assert "a1_scale is None" in source
    assert "a2_scale is None" in source
    assert "if use_w16a16_asm:" in source
    assert "if use_asm:" not in source
    compile(source, "vllm___aiter_ops.patch.py", "exec")


def test_unquantized_moe_asm_shuffle_has_front_loaded_blockers() -> None:
    source = (
        REPO_ROOT
        / "vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py"
    ).read_text()

    expected_checks = [
        "activation is not silu or gelu_tanh",
        "w13_weight or w2_weight is not on CUDA/ROCm device",
        "unsupported w13_weight dtype",
        "unsupported w2_weight dtype",
        "w13_weight / w2_weight shape rank or expert dim mismatch",
        "w13_weight and w2_weight shapes are incompatible",
        "shape alignment requires K % 32 == 0 and N % 16 == 0",
    ]
    for check in expected_checks:
        assert check in source

    assert "_raise_if_aiter_moe_asm_blocked(self, layer)" in source
    assert source.index("_raise_if_aiter_moe_asm_blocked(self, layer)") < source.index(
        "asm_shuffle_weight_b8"
    )
    compile(source, "unquantized_fused_moe_method.py", "exec")
