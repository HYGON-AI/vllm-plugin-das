# SPDX-License-Identifier: Apache-2.0
"""HCU INT8/FP8 MoE quant descriptors and SP all-to-all capability."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.config"
PATCH_ID = "worker.op_opt.moe.config"
TARGETS = (
    f"{TARGET_MODULE}._quant_flags_to_group_shape",
    f"{TARGET_MODULE}.FusedMoEQuantConfig.__post_init__",
    f"{TARGET_MODULE}.FusedMoEQuantConfig.make",
    f"{TARGET_MODULE}.int8_w8a8_moe_quant_config",
    f"{TARGET_MODULE}.FusedMoEParallelConfig.use_all2all_kernels",
    f"{TARGET_MODULE}.FusedMoEParallelConfig.use_deepep_auto_kernels",
    f"{TARGET_MODULE}.FusedMoEParallelConfig.use_batched_activation_format",
    f"{TARGET_MODULE}.FusedMoEParallelConfig.needs_round_robin_routing_tables",
    f"{TARGET_MODULE}.FusedMoEConfig.use_deepep_auto_kernels",
    f"{TARGET_MODULE}.FusedMoEParallelConfig.make",
)
_MARKER = "_vllm_hcu_moe_config_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.fused_moe import config_runtime

    quant_class = require_class(target, "FusedMoEQuantConfig", TARGETS[1].rsplit(".", 1)[0])
    parallel_class = require_class(target, "FusedMoEParallelConfig", TARGETS[4].rsplit(".", 1)[0])
    moe_class = require_class(target, "FusedMoEConfig", TARGETS[8].rsplit(".", 1)[0])
    flags = require_callable(target, "_quant_flags_to_group_shape", TARGETS[0])
    post_init = require_callable(quant_class, "__post_init__", TARGETS[1])
    make = require_callable(quant_class, "make", TARGETS[2])
    int8_config = require_callable(target, "int8_w8a8_moe_quant_config", TARGETS[3])
    all2all_prop = vars(parallel_class).get("use_all2all_kernels")
    if not isinstance(all2all_prop, property) or all2all_prop.fget is None:
        raise PatchCompatibilityError(f"required HCU patch target {TARGETS[4]} is missing")
    require_parameter_names(
        flags,
        TARGETS[0],
        ("quant_dtype", "per_act_token_quant", "per_out_ch_quant", "block_shape"),
    )
    require_parameter_names(post_init, TARGETS[1], ("self",))
    require_parameter_names(
        make,
        TARGETS[2],
        (
            "quant_dtype", "per_act_token_quant", "per_out_ch_quant", "block_shape",
            "w1_scale", "w2_scale", "a1_scale", "a2_scale", "g1_alphas",
            "g2_alphas", "a1_gscale", "a2_gscale", "w1_bias", "w2_bias",
            "w1_zp", "w2_zp", "weight_dtype", "is_scale_swizzled",
            "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit",
        ),
    )
    require_parameter_names(
        int8_config,
        TARGETS[3],
        (
            "w1_scale",
            "w2_scale",
            "a1_scale",
            "a2_scale",
            "w1_bias",
            "w2_bias",
            "per_act_token_quant",
        ),
    )
    require_parameter_names(all2all_prop.fget, TARGETS[4], ("self",))
    batched_prop = vars(parallel_class).get("use_batched_activation_format")
    routing_prop = vars(parallel_class).get("needs_round_robin_routing_tables")
    if not isinstance(batched_prop, property) or batched_prop.fget is None:
        raise PatchCompatibilityError(f"required HCU patch target {TARGETS[6]} is missing")
    if not isinstance(routing_prop, property) or routing_prop.fget is None:
        raise PatchCompatibilityError(f"required HCU patch target {TARGETS[7]} is missing")
    require_parameter_names(batched_prop.fget, TARGETS[6], ("self",))
    require_parameter_names(routing_prop.fget, TARGETS[7], ("self",))
    parallel_make = require_callable(parallel_class, "make", TARGETS[9])
    require_parameter_names(
        parallel_make,
        TARGETS[9],
        ("tp_size_", "pcp_size_", "dp_size_", "sp_size_", "vllm_parallel_config"),
    )

    @functools.wraps(flags)
    def hcu_flags(quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape):
        return config_runtime.quant_flags_to_group_shape(
            target, flags, quant_dtype, per_act_token_quant, per_out_ch_quant, block_shape
        )

    @functools.wraps(post_init)
    def hcu_post_init(self):
        # The normalized HCU descriptor is block-shaped and therefore already
        # passes upstream.  Preserve this guard for manually constructed
        # descriptors while retaining every official validation otherwise.
        if config_runtime.is_hcu_block_quant(target, self.quant_dtype, self.block_shape):
            return None
        return post_init(self)

    @functools.wraps(make)
    def hcu_make(
        quant_dtype=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=None,
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
        is_scale_swizzled=True,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
    ):
        return config_runtime.make_quant_config(
            target, make, quant_dtype, per_act_token_quant, per_out_ch_quant,
            block_shape, w1_scale, w2_scale, a1_scale, a2_scale, g1_alphas,
            g2_alphas, a1_gscale, a2_gscale, w1_bias, w2_bias, w1_zp, w2_zp,
            weight_dtype, is_scale_swizzled, gemm1_alpha, gemm1_beta,
            gemm1_clamp_limit,
        )

    @functools.wraps(int8_config)
    def hcu_int8_config(
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
        block_shape=None,
    ):
        return config_runtime.int8_w8a8_moe_quant_config(
            target, int8_config, w1_scale, w2_scale, a1_scale, a2_scale,
            w1_bias, w2_bias, per_act_token_quant, block_shape,
        )

    del hcu_int8_config.__wrapped__

    @functools.wraps(parallel_make)
    def hcu_parallel_make(
        tp_size_, pcp_size_, dp_size_, sp_size_, vllm_parallel_config
    ):
        result = parallel_make(
            tp_size_,
            pcp_size_,
            dp_size_,
            sp_size_,
            vllm_parallel_config,
        )
        if getattr(vllm_parallel_config, "_vllm_hcu_deepep_auto", False):
            result.all2all_backend = "deepep_auto"
        return result

    target._vllm_hcu_original_quant_flags_to_group_shape = flags
    target._quant_flags_to_group_shape = hcu_flags
    quant_class._vllm_hcu_original_post_init = post_init
    quant_class.__post_init__ = hcu_post_init
    quant_class._vllm_hcu_original_make = make
    quant_class.make = staticmethod(hcu_make)
    target._vllm_hcu_original_int8_w8a8_moe_quant_config = int8_config
    target.int8_w8a8_moe_quant_config = hcu_int8_config
    parallel_class._vllm_hcu_original_use_all2all_kernels = all2all_prop
    parallel_class.use_all2all_kernels = property(config_runtime.use_all2all_kernels)
    parallel_class.use_deepep_auto_kernels = property(
        config_runtime.use_deepep_auto_kernels
    )
    parallel_class._vllm_hcu_original_use_batched_activation_format = batched_prop
    parallel_class.use_batched_activation_format = property(
        lambda self: config_runtime.use_batched_activation_format(
            self, batched_prop
        )
    )
    parallel_class._vllm_hcu_original_needs_round_robin_routing_tables = routing_prop
    parallel_class.needs_round_robin_routing_tables = property(
        lambda self: config_runtime.needs_round_robin_routing_tables(
            self, routing_prop
        )
    )
    moe_class.use_deepep_auto_kernels = property(
        lambda self: self.moe_parallel_config.use_deepep_auto_kernels
    )
    parallel_class._vllm_hcu_original_make = staticmethod(parallel_make)
    parallel_class.make = staticmethod(hcu_parallel_make)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
