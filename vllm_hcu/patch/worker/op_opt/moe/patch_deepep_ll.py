# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepEP low-latency HCU INT8/FP8 dispatch adapter."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll"
PATCH_ID = "worker.op_opt.moe.prepare_finalize.deepep_ll"
TARGETS = (
    f"{TARGET_MODULE}.DeepEPLLPrepareAndFinalize.__init__",
    f"{TARGET_MODULE}.DeepEPLLPrepareAndFinalize._do_quant",
    f"{TARGET_MODULE}.DeepEPLLPrepareAndFinalize.prepare_async",
    f"{TARGET_MODULE}.DeepEPLLPrepareAndFinalize._receiver",
)
_MARKER = "_vllm_hcu_deepep_ll_applied"


def _expose_actual_signature(function):
    # ``functools.wraps`` is useful for diagnostics but the HCU entry points
    # intentionally add parameters, so signature consumers must see them.
    function.__wrapped__ = None
    del function.__wrapped__


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime

    cls = require_class(target, "DeepEPLLPrepareAndFinalize", TARGETS[0].rsplit(".", 1)[0])
    init = require_callable(cls, "__init__", TARGETS[0])
    do_quant = require_callable(cls, "_do_quant", TARGETS[1])
    prepare = require_callable(cls, "prepare_async", TARGETS[2])
    receiver = require_callable(cls, "_receiver", TARGETS[3])
    require_parameter_names(
        init,
        TARGETS[0],
        (
            "self",
            "buffer",
            "max_tokens_per_rank",
            "num_dispatchers",
            "use_fp8_dispatch",
            "global_to_physical",
            "physical_to_global",
            "local_expert_global_ids",
        ),
    )
    require_parameter_names(do_quant, TARGETS[1], ("self", "x", "a1_dtype", "quant_config"))
    require_parameter_names(
        prepare,
        TARGETS[2],
        (
            "self",
            "a1",
            "topk_weights",
            "topk_ids",
            "num_experts",
            "expert_map",
            "apply_router_weight_on_input",
            "quant_config",
            "defer_input_quant",
        ),
    )
    require_parameter_names(
        receiver,
        TARGETS[3],
        (
            "self",
            "expert_x",
            "expert_num_tokens",
            "a1_scale",
            "a1_dtype",
            "quant_config",
        ),
    )

    @functools.wraps(init)
    def hcu_init(
        self,
        buffer,
        max_tokens_per_rank,
        num_dispatchers,
        use_fp8_dispatch=False,
        global_to_physical=None,
        physical_to_global=None,
        local_expert_global_ids=None,
        use_int8_dispatch=False,
    ):
        return deepep_runtime.ll_init(
            init,
            self,
            buffer,
            max_tokens_per_rank,
            num_dispatchers,
            use_fp8_dispatch,
            global_to_physical,
            physical_to_global,
            local_expert_global_ids,
            use_int8_dispatch,
        )

    @functools.wraps(do_quant)
    def hcu_do_quant(
        self,
        x,
        a1_dtype,
        quant_config,
        expert_num_tokens=None,
    ):
        return deepep_runtime.ll_do_quant(
            target,
            self,
            x,
            a1_dtype,
            quant_config,
            expert_num_tokens,
        )

    @functools.wraps(prepare)
    def hcu_prepare_async(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        return deepep_runtime.ll_prepare_async(
            target,
            prepare,
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )

    @functools.wraps(receiver)
    def hcu_receiver(
        self,
        expert_x,
        expert_num_tokens,
        a1_scale,
        a1_dtype,
        quant_config,
    ):
        return deepep_runtime.ll_receiver(
            target,
            self,
            expert_x,
            expert_num_tokens,
            a1_scale,
            a1_dtype,
            quant_config,
        )

    _expose_actual_signature(hcu_init)
    _expose_actual_signature(hcu_do_quant)
    cls._vllm_hcu_original_init = init
    cls.__init__ = hcu_init
    cls._vllm_hcu_original_do_quant = do_quant
    cls._do_quant = hcu_do_quant
    cls._vllm_hcu_original_prepare_async = prepare
    cls.prepare_async = hcu_prepare_async
    cls._vllm_hcu_original_receiver = receiver
    cls._receiver = hcu_receiver
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]


# === K3 LL size logger (v29, replaces earlier probes) ===
import functools as _ft
import json as _json


def _k3_ll_layout_estimate(num_tokens, hidden, num_ranks, num_experts,
                           layered=False, quant_group_size=0):
    NUM_MAX_NVL_PEERS = 8
    num_scales = hidden // 128
    num_nodes = max(1, num_ranks // NUM_MAX_NVL_PEERS)
    msg_d = 16 + max(hidden * 2, hidden + (num_scales if quant_group_size else 4) * 4)
    msg_c = hidden * 2 + (0 if layered else num_scales * 4)
    send = max(num_tokens * msg_d, num_experts * num_tokens * msg_c)
    recv = max(num_experts * num_tokens * msg_d, num_experts * num_tokens * msg_c)
    sig = num_experts * 8
    if layered:
        sig += NUM_MAX_NVL_PEERS * num_nodes * num_tokens * 4 + NUM_MAX_NVL_PEERS * 4
    return (send + recv + sig) * 2


def _install_k3_ll_logger():
    import deep_ep as _dep

    if getattr(_dep.Buffer, "_k3_ll_size_logger", False):
        return
    _orig_init = _dep.Buffer.__init__

    @_ft.wraps(_orig_init)
    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            with open("/tmp/k3_ll_sizes.log", "a") as f:
                f.write(_json.dumps({
                    "event": "ctor",
                    "num_rdma_bytes": getattr(self, "num_rdma_bytes", None),
                    "num_nvl_bytes": getattr(self, "num_nvl_bytes", None),
                    "kwargs": {k: str(v)[:48] for k, v in kw.items()},
                }) + "\n")
        except Exception:
            pass

    _orig_disp = _dep.Buffer.low_latency_dispatch

    @_ft.wraps(_orig_disp)
    def _disp(self, x, topk_idx, topk_weight=None, num_max=0, num_experts=0, **kw):
        try:
            hidden = int(x.shape[-1]) if hasattr(x, "shape") and x.dim() >= 2 else 0
            est = _k3_ll_layout_estimate(
                int(num_max), hidden, int(self.group_size), int(num_experts),
                layered=False, quant_group_size=int(kw.get("quant_group_size") or 0))
        except Exception:
            est = None
        rb = getattr(self, "num_rdma_bytes", None)
        with open("/tmp/k3_ll_sizes.log", "a") as f:
            f.write(_json.dumps({
                "event": "dispatch",
                "x_shape": list(x.shape) if hasattr(x, "shape") else None,
                "num_max_tokens": int(num_max),
                "num_experts": int(num_experts),
                "buffer_num_rdma_bytes": rb,
                "layout_formula_total_bytes": est,
                "ratio_formula_over_buffer": (round(est / rb, 4) if est and rb else None),
                "quant_group_size": kw.get("quant_group_size"),
            }) + "\n")
        return _orig_disp(self, x, topk_idx, topk_weight, num_max, num_experts, **kw)

    _dep.Buffer.__init__ = _init
    _dep.Buffer.low_latency_dispatch = _disp
    _dep.Buffer._k3_ll_size_logger = True


_install_k3_ll_logger()
