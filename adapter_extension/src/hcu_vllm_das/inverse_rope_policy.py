# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Plugin entry point for the shared native PCP RoPE selection policy."""
from hcu_adapter.kernels.pcp_rope_policy import selected_mode as rope_mode


def selected_mode():
    return rope_mode("vllm_inverse_rope")
