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

TARGET_MODULE = "vllm.distributed.eplb.eplb_state"
PATCH_ID = "worker.framework_opt.step5_eplb_state"
TARGETS = (
    "vllm.distributed.eplb.eplb_state.EplbState.step",
    "vllm.distributed.eplb.eplb_state.EplbState._propagate_shared_tensors",
    "vllm.distributed.eplb.eplb_state.EplbState.rearrange",
    "vllm.distributed.eplb.eplb_state._move_to_workspace",
    "vllm.distributed.eplb.eplb_state.create_eplb_communicator",
)
_MARKER = "_vllm_hcu_step5_eplb_state_applied"

SIGNATURES = (
    (TARGETS[3], ("model_state", "ep_rank")),
    (TARGETS[4], ("group_coordinator", "backend", "expert_weights", "expert_buffer")),
    (
        "vllm.distributed.eplb.eplb_state.EplbState.step",
        ("self", "is_dummy", "is_profile", "log_stats"),
    ),
    (
        "vllm.distributed.eplb.eplb_state.EplbState._propagate_shared_tensors",
        ("self", "model", "num_unpadded_tokens_tensors"),
    ),
    (
        "vllm.distributed.eplb.eplb_state.EplbState.rearrange",
        ("self", "is_profile", "rank_mapping"),
    ),
)
REVIEWED_SOURCES = (
    (
        TARGETS[4],
        ("40072a27544d91da0fde0eedec61ba3686fd2726457397110b387581bcd051c3",),
    ),
    (
        TARGETS[3],
        ("4d024a8b1332524acc9e6af21e71c39aea92915a92811c8d512b6db04d0f8d23",),
    ),
    (
        "vllm.distributed.eplb.eplb_state.EplbState.step",
        (
            "7edbc15adab6c01ce2b4b28549cc46f08b1a666d563ed48e8be11fdab5ea3eaa",
            "b6408e5f470d6bd44abf7382fbb4be00f2e2c69d1c9e44fb49580fd43edbffc0",
        ),
    ),
    (
        "vllm.distributed.eplb.eplb_state.EplbState._propagate_shared_tensors",
        (
            "575bba89662116c26b629554a9d41d8b3c502b944c5ca7e2e53fdfb7a39bee28",
            "f22c6c3dc27dd76223570eb793fb5fb40e157b00d2b2c5ce80041d1cdb180396",
        ),
    ),
    (
        "vllm.distributed.eplb.eplb_state.EplbState.rearrange",
        (
            "159255fed5e5e0ce282d731ad80292f528963b0fe272b64b35a93a1a918796c4",
            "48864a17493f250b101fde1b90f22ef4da6d71f1ffc9f6cafaf5da05cb1681f3",
        ),
    ),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    cls = getattr(target, "EplbState", None)
    for name, symbol in zip(
        ("step", "_propagate_shared_tensors", "rearrange"), TARGETS[:3]
    ):
        require_callable(cls, name, symbol)
    require_callable(target, "_move_to_workspace", TARGETS[3])
    require_callable(target, "create_eplb_communicator", TARGETS[4])
    from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
    from vllm_hcu.distributed import eplb_runtime

    cls.step = eplb_runtime.step
    cls._propagate_shared_tensors = eplb_runtime._propagate_shared_tensors
    cls.rearrange = eplb_runtime.rearrange
    target._move_to_workspace = eplb_runtime._move_to_workspace
    target.create_eplb_communicator = create_eplb_communicator
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
