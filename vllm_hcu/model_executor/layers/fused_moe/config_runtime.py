# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""vLLM v0.25.1 target MoE configuration behavior plus HCU quantization deltas.

The functions in this module intentionally receive the imported vLLM module
as an argument.  This keeps import-time dependencies light and, more
importantly, makes every upstream symbol used by the implementation explicit
and auditable.
"""

from __future__ import annotations

def is_hcu_block_quant(module: object, quant_dtype: object, block_shape: object) -> bool:
    """Return whether HCU supports the otherwise-conflicting block layout."""

    if block_shape is None:
        return False
    return quant_dtype == module.torch.int8 or quant_dtype == module.current_platform.fp8_dtype()


def quant_flags_to_group_shape(
    module: object,
    original: object,
    quant_dtype: object,
    per_act_token_quant: bool,
    per_out_ch_quant: bool,
    block_shape: list[int] | None,
):
    if not is_hcu_block_quant(module, quant_dtype, block_shape):
        return original(
            quant_dtype,
            per_act_token_quant,
            per_out_ch_quant,
            block_shape,
        )

    # HCU INT8/FP8 DeepEP kernels carry a block-shaped activation descriptor
    # even when the checkpoint described activation quantization as per-token.
    # Upstream's two flags cannot express that combination, so the sidecar
    # adapter normalizes both descriptors to the audited block shape.
    assert block_shape is not None
    group_shape = module.GroupShape(row=block_shape[0], col=block_shape[1])
    return group_shape, group_shape


def make_quant_config(
    module: object,
    original: object,
    quant_dtype=None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    g1_alphas=None,
    g2_alphas=None,
    a1_gscale=None,
    a2_gscale=None,
    w1_bias=None,
    w2_bias=None,
    w1_zp=None,
    w2_zp=None,
    weight_dtype=None,
    is_scale_swizzled: bool = True,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
):
    special = is_hcu_block_quant(module, quant_dtype, block_shape)
    # Calling upstream with the normalized flags preserves its descriptor
    # construction and every future field while avoiding only the two legacy
    # assertions that cannot represent HCU's block-dispatch contract.
    return original(
        quant_dtype,
        False if special else per_act_token_quant,
        False if special else per_out_ch_quant,
        block_shape,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        g1_alphas,
        g2_alphas,
        a1_gscale,
        a2_gscale,
        w1_bias,
        w2_bias,
        w1_zp,
        w2_zp,
        weight_dtype,
        is_scale_swizzled,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
    )


def int8_w8a8_moe_quant_config(
    module: object,
    original: object,
    w1_scale,
    w2_scale,
    a1_scale,
    a2_scale,
    w1_bias=None,
    w2_bias=None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
    gemm1_clamp_limit: float | None = None,
):
    if block_shape is None and gemm1_clamp_limit is None:
        return original(
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            w1_bias,
            w2_bias,
            per_act_token_quant,
        )
    return module.FusedMoEQuantConfig.make(
        module.torch.int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=False,
        block_shape=block_shape,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def use_all2all_kernels(parallel_config: object) -> bool:
    return bool(
        (parallel_config.dp_size > 1 or parallel_config.is_sequence_parallel)
        and parallel_config.use_ep
    )


def use_deepep_auto_kernels(parallel_config: object) -> bool:
    return bool(
        use_all2all_kernels(parallel_config)
        and parallel_config.all2all_backend == "deepep_auto"
    )


def use_batched_activation_format(
    parallel_config: object, original_property: property
) -> bool:
    assert original_property.fget is not None
    return bool(
        use_deepep_auto_kernels(parallel_config)
        or original_property.fget(parallel_config)
    )


def needs_round_robin_routing_tables(
    parallel_config: object, original_property: property
) -> bool:
    assert original_property.fget is not None
    return bool(
        use_deepep_auto_kernels(parallel_config)
        or original_property.fget(parallel_config)
    )


__all__ = [
    "int8_w8a8_moe_quant_config",
    "is_hcu_block_quant",
    "make_quant_config",
    "quant_flags_to_group_shape",
    "use_all2all_kernels",
    "use_batched_activation_format",
    "use_deepep_auto_kernels",
    "needs_round_robin_routing_tables",
]
