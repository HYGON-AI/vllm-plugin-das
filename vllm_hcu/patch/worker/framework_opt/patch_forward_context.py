# SPDX-License-Identifier: Apache-2.0
"""Forward-context adapter preserving vLLM's official dataclass identity."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.forward_context"
PATCH_ID = "worker.framework_opt.forward_context.hcu_runtime_fields"
TARGETS = (
    f"{TARGET_MODULE}.set_forward_context",
    f"{TARGET_MODULE}.ForwardContext",
    f"{TARGET_MODULE}.create_forward_context",
)
_MARKER = "_vllm_hcu_forward_context_applied"
_WRAPPER = "_vllm_hcu_forward_context_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    forward = load_exact_module(TARGET_MODULE, module)
    context_class = require_class(forward, "ForwardContext", TARGETS[1])
    wrapped = (
        (forward, "create_forward_context", TARGETS[2], _WRAPPER),
        (forward, "set_forward_context", TARGETS[0], _WRAPPER),
    )
    if already_applied(forward, _MARKER, wrapped):
        return False

    # The class must stay the exact official dataclass.  A slots dataclass
    # would not permit the required per-forward runtime attributes.
    if "__dataclass_fields__" not in vars(context_class) or "__slots__" in vars(
        context_class
    ):
        raise PatchCompatibilityError(
            f"required target {TARGETS[1]} is not the extensible v0.25 dataclass"
        )
    original_create = require_callable(forward, "create_forward_context", TARGETS[2])
    create_positional = (
        "attn_metadata",
        "vllm_config",
        "dp_metadata",
        "cudagraph_runtime_mode",
        "batch_descriptor",
        "ubatch_slices",
        "slot_mapping",
        "additional_kwargs",
        "skip_compiled",
        "is_padding",
    )
    create_defaults = {
        "dp_metadata": None,
        "cudagraph_runtime_mode": forward.CUDAGraphMode.NONE,
        "batch_descriptor": None,
        "ubatch_slices": None,
        "slot_mapping": None,
        "additional_kwargs": None,
        "skip_compiled": False,
        "is_padding": None,
    }
    require_exact_signature(
        original_create,
        TARGETS[2],
        positional=create_positional,
        defaults=create_defaults,
    )
    original_set = require_callable(forward, "set_forward_context", TARGETS[0])
    set_positional = (
        "attn_metadata",
        "vllm_config",
        "num_tokens",
        "num_tokens_across_dp",
        "cudagraph_runtime_mode",
        "batch_descriptor",
        "ubatch_slices",
        "slot_mapping",
        "skip_compiled",
        "is_padding",
    )
    set_defaults = {
        "num_tokens": None,
        "num_tokens_across_dp": None,
        "cudagraph_runtime_mode": forward.CUDAGraphMode.NONE,
        "batch_descriptor": None,
        "ubatch_slices": None,
        "slot_mapping": None,
        "skip_compiled": False,
        "is_padding": None,
    }
    require_exact_signature(
        original_set,
        TARGETS[0],
        positional=set_positional,
        defaults=set_defaults,
    )

    from vllm_hcu import forward_context_runtime

    @functools.wraps(original_create)
    def hcu_create(
        attn_metadata,
        vllm_config,
        dp_metadata=None,
        cudagraph_runtime_mode=forward.CUDAGraphMode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        slot_mapping=None,
        additional_kwargs=None,
        skip_compiled=False,
        is_padding=None,
        *,
        scatter_indexes_tensor=None,
        gather_indexes_tensor=None,
        enable_lightly_cp=False,
        enable_lightly_cplb=False,
    ):
        context = original_create(
            attn_metadata,
            vllm_config,
            dp_metadata,
            cudagraph_runtime_mode,
            batch_descriptor,
            ubatch_slices,
            slot_mapping,
            additional_kwargs,
            skip_compiled,
            is_padding,
        )
        return forward_context_runtime.attach_hcu_context_fields(
            context,
            scatter_indexes_tensor=scatter_indexes_tensor,
            gather_indexes_tensor=gather_indexes_tensor,
            enable_lightly_cp=enable_lightly_cp,
            enable_lightly_cplb=enable_lightly_cplb,
        )

    @functools.wraps(original_set)
    def hcu_set(
        attn_metadata,
        vllm_config,
        num_tokens=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=forward.CUDAGraphMode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        slot_mapping=None,
        skip_compiled=False,
        is_padding=None,
        *,
        scatter_indexes_tensor=None,
        gather_indexes_tensor=None,
        enable_lightly_cp=False,
        enable_lightly_cplb=False,
    ):
        low_latency = (
            vllm_config.parallel_config.all2all_backend == "deepep_low_latency"
        )
        has_hcu_context = bool(
            enable_lightly_cp
            or enable_lightly_cplb
            or scatter_indexes_tensor is not None
            or gather_indexes_tensor is not None
        )
        if not low_latency and not has_hcu_context:
            return original_set(
                attn_metadata,
                vllm_config,
                num_tokens,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                batch_descriptor,
                ubatch_slices,
                slot_mapping,
                skip_compiled,
                is_padding,
            )
        return forward_context_runtime.set_forward_context(
            forward,
            attn_metadata,
            vllm_config,
            num_tokens,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            batch_descriptor,
            ubatch_slices,
            slot_mapping,
            skip_compiled,
            is_padding,
            scatter_indexes_tensor=scatter_indexes_tensor,
            gather_indexes_tensor=gather_indexes_tensor,
            enable_lightly_cp=enable_lightly_cp,
            enable_lightly_cplb=enable_lightly_cplb,
        )

    setattr(hcu_create, _WRAPPER, True)
    setattr(hcu_set, _WRAPPER, True)
    setattr(forward, "_vllm_hcu_original_create_forward_context", original_create)
    setattr(forward, "_vllm_hcu_original_set_forward_context", original_set)
    setattr(forward, "create_forward_context", hcu_create)
    setattr(forward, "set_forward_context", hcu_set)
    setattr(forward, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
