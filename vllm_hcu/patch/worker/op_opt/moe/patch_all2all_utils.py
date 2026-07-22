# SPDX-License-Identifier: Apache-2.0
"""DeepEP LL FP8/INT8 dispatch selection without source mutation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    require_callable,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.all2all_utils"
PATCH_ID = "worker.op_opt.moe.all2all_utils"
TARGETS = (f"{TARGET_MODULE}.maybe_make_prepare_finalize",)
_MARKER = "_vllm_hcu_all2all_dispatch_applied"
_WRAPPER = "_vllm_hcu_all2all_dispatch_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if check_module_marker(target, _MARKER, ((target, "maybe_make_prepare_finalize", _WRAPPER),)):
        return False
    original = require_callable(target, "maybe_make_prepare_finalize", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "moe",
            "quant_config",
            "routing_tables",
            "allow_new_interface",
            "use_monolithic",
            "eep_stage",
        ),
    )

    @functools.wraps(original)
    def hcu_maybe_make_prepare_finalize(
        moe,
        quant_config,
        routing_tables=None,
        allow_new_interface=False,
        use_monolithic=False,
        eep_stage=False,
    ):
        prepare_finalize = original(
            moe,
            quant_config,
            routing_tables,
            allow_new_interface,
            use_monolithic,
            eep_stage,
        )
        ll_class = getattr(target, "DeepEPLLPrepareAndFinalize", None)
        if ll_class is None or not isinstance(prepare_finalize, ll_class):
            return prepare_finalize
        if quant_config is None:
            raise RuntimeError("DeepEP LL requires a FusedMoEQuantConfig")
        use_fp8 = quant_config.quant_dtype == target.current_platform.fp8_dtype()
        use_int8 = quant_config.quant_dtype == target.torch.int8
        if use_fp8 and use_int8:
            raise RuntimeError("DeepEP LL cannot enable FP8 and INT8 dispatch together")
        prepare_finalize.use_fp8_dispatch = use_fp8
        prepare_finalize.use_int8_dispatch = use_int8
        return prepare_finalize

    setattr(hcu_maybe_make_prepare_finalize, _WRAPPER, True)
    target._vllm_hcu_original_maybe_make_prepare_finalize = original
    target.maybe_make_prepare_finalize = hcu_maybe_make_prepare_finalize
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
