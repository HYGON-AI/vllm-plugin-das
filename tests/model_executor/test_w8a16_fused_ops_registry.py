# SPDX-License-Identifier: Apache-2.0
"""Registry inventory for the W8A16 + Qwen3.5 fused-op adapter stack.

References every adapter added by the v0.28.1 W8A16/fused-op port (the
structural coverage gate in tools/check_patch_test_coverage.py requires each
patch_*.py stem to appear in a test) and asserts the shared adapter contract
plus worker registration.
"""

from __future__ import annotations

import importlib

PORTED_ADAPTERS = (
    # compressed-tensors pack-quantized int8 -> HCU W8A16 GEMM routing.
    "patch_compressed_tensors_w8a16_hcu",
    # ModelOpt W8A16 (int8 weight-only) linear + MoE via LINEAR_METHOD_BUILDERS.
    "patch_modelopt_w8a16",
    # Fused rearrange_mixed_qkv + strided-z output projection (GDN path).
    "patch_qwen_gdn_ops",
    # Fused shared-expert sigmoid-mul gate.
    "patch_qwen2_moe",
    # Stride-safe fused RMSNorm + MRoPE (vllm-hcu 4e01bc8).
    "patch_qwen35_mrope",
)


def test_ported_adapters_expose_contract_and_are_registered():
    import vllm_hcu.patch.worker as worker

    registered = [
        spec.adapter
        for group in (
            worker._MOE_FOUNDATION_CALLBACKS,
            worker._CORE_CALLBACKS,
            worker._OP_CALLBACKS,
            worker._MOE_RUNTIME_CALLBACKS,
            worker._FRAMEWORK_CALLBACKS,
        )
        for spec in group
    ]
    for stem in PORTED_ADAPTERS:
        module = importlib.import_module(
            f"vllm_hcu.patch.worker.op_opt.{stem}"
        )
        for symbol in ("PATCH_ID", "TARGET_MODULE", "TARGETS",
                       "apply_to_module"):
            assert hasattr(module, symbol), f"{stem} missing {symbol}"
        assert any(name.endswith(stem) for name in registered), (
            f"{stem} is not armed by the worker dispatcher"
        )
