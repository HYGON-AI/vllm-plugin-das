# SPDX-License-Identifier: Apache-2.0
"""HCU DeepEP runtime tuning without rewriting vLLM's all2all module."""

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

TARGET_MODULE = "vllm.distributed.device_communicators.all2all"
PATCH_ID = "worker.framework_opt.communicator.deep_ep_runtime"
TARGETS = (
    f"{TARGET_MODULE}.DeepEPAll2AllManagerBase.__init__",
    f"{TARGET_MODULE}.DeepEPHTAll2AllManager._make_all2all_kwargs",
    f"{TARGET_MODULE}.DeepEPHTAll2AllManager.set_num_sms",
    f"{TARGET_MODULE}.DeepEPAutoAll2AllManager",
)
_MARKER = "_vllm_hcu_deep_ep_runtime_applied"
_WRAPPER = "_vllm_hcu_deep_ep_runtime_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    all2all = load_exact_module(TARGET_MODULE, module)
    base = require_class(
        all2all, "DeepEPAll2AllManagerBase", f"{TARGET_MODULE}.DeepEPAll2AllManagerBase"
    )
    ht = require_class(
        all2all, "DeepEPHTAll2AllManager", f"{TARGET_MODULE}.DeepEPHTAll2AllManager"
    )
    wrapped = (
        (base, "__init__", TARGETS[0], _WRAPPER),
        (ht, "_make_all2all_kwargs", TARGETS[1], _WRAPPER),
        (ht, "set_num_sms", TARGETS[2], _WRAPPER),
    )
    if already_applied(all2all, _MARKER, wrapped):
        auto = getattr(all2all, "DeepEPAutoAll2AllManager", None)
        if not isinstance(auto, type) or not getattr(
            auto, "is_deepep_auto_manager", False
        ):
            raise RuntimeError("HCU DeepEP auto manager marker is stale")
        return False

    original_init = require_callable(base, "__init__", TARGETS[0])
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "cpu_group", "tcp_store_group"),
        defaults={"tcp_store_group": None},
    )
    original_kwargs = require_callable(ht, "_make_all2all_kwargs", TARGETS[1])
    require_exact_signature(original_kwargs, TARGETS[1], positional=("self",))
    original_set_sms = require_callable(ht, "set_num_sms", TARGETS[2])
    require_exact_signature(
        original_set_sms, TARGETS[2], positional=("self", "num_sms")
    )

    from vllm_hcu.distributed.device_communicators import framework_runtime

    @functools.wraps(original_init)
    def hcu_init(self, cpu_group, tcp_store_group=None):
        original_init(self, cpu_group, tcp_store_group)
        framework_runtime.initialize_deep_ep_manager(self)

    @functools.wraps(original_kwargs)
    def hcu_make_kwargs(self):
        return framework_runtime.make_deep_ep_ht_kwargs(self, all2all.envs)

    @functools.wraps(original_set_sms)
    def hcu_set_num_sms(self, num_sms):
        return original_set_sms(
            self, framework_runtime.requested_deep_ep_sms(num_sms)
        )

    for function in (hcu_init, hcu_make_kwargs, hcu_set_num_sms):
        setattr(function, _WRAPPER, True)
    setattr(base, "_vllm_hcu_original_init", original_init)
    setattr(ht, "_vllm_hcu_original_make_all2all_kwargs", original_kwargs)
    setattr(ht, "_vllm_hcu_original_set_num_sms", original_set_sms)
    setattr(base, "__init__", hcu_init)
    setattr(ht, "_make_all2all_kwargs", hcu_make_kwargs)
    setattr(ht, "set_num_sms", hcu_set_num_sms)
    framework_runtime.install_deep_ep_auto_manager(all2all)
    setattr(all2all, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
