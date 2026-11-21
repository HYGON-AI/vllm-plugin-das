# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Plugin entry point for the shared native PCP RoPE selection policy."""
from .adapter_api import adapter_module, optional_adapter_module


def selected_mode():
    policy = optional_adapter_module("hcu_adapter.kernels.pcp_rope_policy")
    if policy is not None:
        return policy.selected_mode("vllm_inverse_rope")

    op_disabled = adapter_module("kill_switch").op_disabled
    if op_disabled("vllm_pcp") or op_disabled("vllm_inverse_rope"):
        return "stock"
    mode = adapter_module("hcu_config").load()["ops"].get("vllm_inverse_rope", "stock")
    if mode not in ("stock", "hcu"):
        raise ValueError("ops.vllm_inverse_rope must be stock or hcu")
    return mode
