# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Resolve Qwen4Exp Engram settings with HCU compatibility fallbacks."""

from __future__ import annotations

from vllm.config import get_current_vllm_config_or_none

from vllm_hcu.platforms import envs as hcu_envs


def cpu_offload_enabled() -> bool:
    """Prefer EngramConfig and fall back to the legacy HCU environment flag."""
    vllm_config = get_current_vllm_config_or_none()
    engram_config = (
        getattr(vllm_config, "engram_config", None)
        if vllm_config is not None
        else None
    )
    if engram_config is not None:
        return bool(engram_config.cpu_offload)
    return bool(hcu_envs.VLLM_HCU_PLE_CPU_OFFLOAD)


__all__ = ["cpu_offload_enabled"]
