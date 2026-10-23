# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lightweight ownership marker for HY4-only quantization routes."""

from __future__ import annotations

HYV4_MODEL_MARKER = "_vllm_hcu_hyv4_model"


def mark_hyv4_quant_config(quant_config):
    if quant_config is not None:
        setattr(quant_config, HYV4_MODEL_MARKER, True)
    return quant_config


__all__ = ["HYV4_MODEL_MARKER", "mark_hyv4_quant_config"]
