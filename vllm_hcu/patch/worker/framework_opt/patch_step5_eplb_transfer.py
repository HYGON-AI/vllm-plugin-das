# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.distributed.eplb.rebalance_execute"
PATCH_ID = "worker.framework_opt.step5_eplb_transfer"
TARGETS = (
    "vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace",
    "vllm.distributed.eplb.rebalance_execute.transfer_layer",
)
_MARKER = "_vllm_hcu_step5_eplb_transfer_applied"

SIGNATURES = (
    (
        TARGETS[1],
        (
            "old_layer_indices", "new_layer_indices", "expert_weights",
            "expert_weights_buffer", "ep_group", "communicator", "is_profile",
            "cuda_stream", "rank_mapping", "layer_idx",
        ),
    ),
    (
        "vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace",
        (
            "old_global_expert_indices",
            "new_global_expert_indices",
            "expert_weights",
            "expert_buffer",
            "ep_group",
            "communicator",
            "is_profile",
            "rank_mapping",
        ),
    ),
)
REVIEWED_SOURCES = (
    (
        TARGETS[1],
        ("ae3aa7ab3c8152ab1bd328c4651d54297dca6426cd672ef9e13a2ffd7ac93cd7",),
    ),
    (
        "vllm.distributed.eplb.rebalance_execute.rearrange_expert_weights_inplace",
        (
            "a9b7a65933aeb43fc7837983037e633d8da813edd28c2c8ef24037903b9cdd76",
            "ce2dc128c05aaa4dc008d21a86ab84e93ec2d43bfba0c7492d5eecce80002653",
        ),
    ),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    require_callable(target, "rearrange_expert_weights_inplace", TARGETS[0])
    require_callable(target, "transfer_layer", TARGETS[1])
    from vllm_hcu.distributed.eplb_transfer_runtime import (
        rearrange_expert_weights_inplace,
        transfer_layer,
    )

    target.rearrange_expert_weights_inplace = rearrange_expert_weights_inplace
    target.transfer_layer = transfer_layer
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
