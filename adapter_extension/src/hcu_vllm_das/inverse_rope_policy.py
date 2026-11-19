# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free configuration policy for optional PCP inverse RoPE."""
from hcu_adapter import hcu_config
from hcu_adapter.kill_switch import op_disabled


def selected_mode():
    if op_disabled("vllm_pcp") or op_disabled("vllm_inverse_rope"):
        return "stock"
    mode = hcu_config.load()["ops"].get("vllm_inverse_rope", "stock")
    if mode not in ("stock", "hcu"):
        raise ValueError("ops.vllm_inverse_rope must be stock or hcu")
    return mode
