# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable DeepEP all2all under PCP + EP even when DP=TP=1.

Upstream vLLM (base_device_communicator.DeviceCommunicatorBase.__init__)
computes ``use_all2all = is_ep_communicator and (dp_size > 1 or
use_sequence_parallel_moe)``. In the HCU PCP + EP scenario with DP=TP=1,
this evaluates to False, so cuda_communicator never constructs a
DeepEP all2all manager. This adapter widens the gate to also cover
PCP > 1 + EP + explicit DeepEP backend.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.base_device_communicator"
PATCH_ID = "worker.framework_opt.communicator.base_pcp_ep"
TARGETS = (f"{TARGET_MODULE}.DeviceCommunicatorBase.__init__",)
_MARKER = "_vllm_hcu_base_pcp_ep_applied"
_WRAPPER = "_vllm_hcu_base_pcp_ep_wrapper"

_DEEPEP_BACKENDS = frozenset(
    {"deepep_high_throughput", "deepep_low_latency"}
)


def apply_to_module(module: ModuleType) -> bool:
    base_mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        base_mod,
        "DeviceCommunicatorBase",
        f"{TARGET_MODULE}.DeviceCommunicatorBase",
    )
    wrapped = ((cls, "__init__", TARGETS[0], _WRAPPER),)
    if already_applied(base_mod, _MARKER, wrapped):
        return False

    original_init = require_callable(cls, "__init__", TARGETS[0])
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=(
            "self",
            "cpu_group",
            "device",
            "device_group",
            "unique_name",
            "global_ranks",
            "global_world_size",
        ),
        defaults={
            "device": None,
            "device_group": None,
            "unique_name": "",
            "global_ranks": None,
            "global_world_size": None,
        },
    )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
    ):
        original_init(
            self,
            cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
        )
        # Upstream already enabled all2all via DP>1 / use_sequence_parallel_moe.
        if getattr(self, "use_all2all", False):
            return
        # Only EP communicators care about all2all managers.
        if not getattr(self, "is_ep_communicator", False):
            return

        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        if cfg is None:
            return
        pc = getattr(cfg, "parallel_config", None)
        if pc is None:
            return
        pcp_size = int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
        ep_enabled = bool(getattr(pc, "enable_expert_parallel", False))
        backend = getattr(pc, "all2all_backend", None)
        if (
            pcp_size > 1
            and ep_enabled
            and backend in _DEEPEP_BACKENDS
        ):
            self.use_all2all = True
            print(
                "[HCU PATCH base_pcp_ep] force use_all2all=True "
                f"pcp={pcp_size} backend={backend} unique_name={unique_name!r}",
                flush=True,
            )

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "__init__", hcu_init)
    setattr(base_mod, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
