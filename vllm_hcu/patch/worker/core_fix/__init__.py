# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model compatibility fixes required in worker processes.

This package deliberately performs no import-time patching.  Each module
exports ``apply_to_module(module)`` for an exact import callback and an
explicit ``apply(module=None)`` convenience entry point.
"""

from . import (
    patch_deepseek_v32_config,
    patch_gpt_oss_mlp_block,
    patch_qwen3_5_mamba_state_dtype,
    patch_qwen3_vl,
    patch_qwen3_vl_moe,
)

__all__ = [
    "patch_deepseek_v32_config",
    "patch_gpt_oss_mlp_block",
    "patch_qwen3_5_mamba_state_dtype",
    "patch_qwen3_vl",
    "patch_qwen3_vl_moe",
]
