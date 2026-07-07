# SPDX-License-Identifier: Apache-2.0

"""
Patch FP8 MoE oracle to expose the DeepEP DeepGEMM backend.
"""

PATCHES = [
(
"""
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
""",
"""
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    DPSK_DEEPGEMM = "DPSK_DEEPGEMM"
    BATCHED_TRITON = "BATCHED_TRITON"
""",
),

(
"""
    elif backend == Fp8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
""",
"""
    elif backend == Fp8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == Fp8MoeBackend.DPSK_DEEPGEMM:
        from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
            DeepEPDeepGemmContiguousExperts,
            DeepEPDeepGemmMaskedExperts,
        )

        return [DeepEPDeepGemmContiguousExperts, DeepEPDeepGemmMaskedExperts]

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
""",
),

(
"""
    mapping = {
        "triton": Fp8MoeBackend.TRITON,
        "deep_gemm": Fp8MoeBackend.DEEPGEMM,
""",
"""
    mapping = {
        "triton": Fp8MoeBackend.TRITON,
        "dpsk_deep_gemm": Fp8MoeBackend.DPSK_DEEPGEMM,
        "deep_gemm": Fp8MoeBackend.DEEPGEMM,
""",
),

(
"""
        if fp8_backend not in [
            Fp8MoeBackend.TRITON,
            Fp8MoeBackend.BATCHED_TRITON,
""",
"""
        if fp8_backend not in [
            Fp8MoeBackend.TRITON,
            Fp8MoeBackend.DPSK_DEEPGEMM,
            Fp8MoeBackend.BATCHED_TRITON,
""",
),

(
"""
    if fp8_backend in [Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM]:
        assert block_quant
        w13, w2, w13_scale, w2_scale = prepare_fp8_moe_layer_for_deepgemm(
            w13,
            w2,
            w13_scale,
            w2_scale,
            tuple(layer.weight_block_size),
        )
""",
"""
    if fp8_backend in [Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM]:
        if block_quant:
            w13, w2, w13_scale, w2_scale = prepare_fp8_moe_layer_for_deepgemm(
                w13,
                w2,
                w13_scale,
                w2_scale,
                tuple(layer.weight_block_size),
            )
""",
),

(
"""
    block_quant = hasattr(layer, "weight_block_size")
""",
"""
    block_quant = getattr(layer, "weight_block_size", None) is not None
""",
),
]
