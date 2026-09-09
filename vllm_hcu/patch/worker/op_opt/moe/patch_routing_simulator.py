# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register deterministic rank-balanced EPLB routing simulation."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.fused_moe.router.routing_simulator_router"
)
PATCH_ID = "worker.op_opt.moe.router.routing_simulator"
TARGETS = (
    f"{TARGET_MODULE}.RoutingStrategy",
    f"{TARGET_MODULE}.RoutingSimulator.register_strategy",
)
STRATEGY_NAME = "hcu_eplb_balancedness"
_MARKER = "_vllm_hcu_eplb_balancedness_strategy_registered"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    simulator = require_class(target, "RoutingSimulator", TARGETS[1])
    available = require_callable(
        simulator,
        "get_available_strategies",
        f"{TARGET_MODULE}.RoutingSimulator.get_available_strategies",
    )
    if getattr(target, _MARKER, False):
        if STRATEGY_NAME not in available():
            raise PatchCompatibilityError(
                f"stale HCU routing simulator marker for {TARGET_MODULE}; "
                "restart process"
            )
        return False

    base = require_class(target, "RoutingStrategy", TARGETS[0])
    register = require_callable(simulator, "register_strategy", TARGETS[1])
    require_parameter_names(register, TARGETS[1], ("name", "strategy"))

    from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
        make_hcu_eplb_balancedness_strategy,
    )

    strategy_type = make_hcu_eplb_balancedness_strategy(base)
    register(STRATEGY_NAME, strategy_type())
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "STRATEGY_NAME",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
