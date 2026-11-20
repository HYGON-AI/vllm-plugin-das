# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free configuration policy for optional PCP inverse RoPE."""

from .adapter_api import adapter_module as _adapter_module
hcu_config = _adapter_module('hcu_config')
op_disabled = _adapter_module('kill_switch').op_disabled


def selected_mode():
    if op_disabled("vllm_pcp") or op_disabled("vllm_inverse_rope"):
        return "stock"
    mode = hcu_config.load()["ops"].get("vllm_inverse_rope", "stock")
    if mode not in ("stock", "hcu"):
        raise ValueError("ops.vllm_inverse_rope must be stock or hcu")
    return mode
