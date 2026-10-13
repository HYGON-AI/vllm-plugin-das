# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the opt-in Gemma4 attention implementation at model construction."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)
_ENV_NAME = "VLLM_HCU_GEMMA4_USE_FA"
_TARGET_ARCHITECTURES = {
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
}


def gemma4_fa_enabled() -> bool:
    raw = os.environ.get(_ENV_NAME, "0").strip().lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(f"{_ENV_NAME} must be 0/1 or false/true; got {raw!r}")
    if "HCU_FLASH_ATTN_Q5" in os.environ:
        logger.warning_once(
            "HCU_FLASH_ATTN_Q5 is no longer used; select the Gemma4 FA path "
            "with VLLM_HCU_GEMMA4_USE_FA instead."
        )
    return raw in {"1", "true"}


def select_gemma4_hcu_attention_backend(
    *, head_dim: int, is_sliding: bool
) -> type[AttentionBackend] | None:
    if not gemma4_fa_enabled():
        logger.info_once("Gemma4 FA fast path disabled; using vLLM backend selection.")
        return None
    if is_sliding or head_dim != 512:
        return None

    from vllm.config import get_current_vllm_config
    from vllm.platforms import current_platform

    if current_platform.device_name != "hip":
        return None
    config = get_current_vllm_config()
    architectures = set(config.model_config.architectures or ())
    if not architectures.intersection(_TARGET_ARCHITECTURES):
        return None
    selected = config.attention_config.backend
    selected_name = getattr(selected, "name", selected)
    if selected_name not in (None, "TRITON_ATTN"):
        raise ValueError(
            f"{_ENV_NAME}=1 requires the Triton-compatible Gemma4 cache and "
            f"metadata; it conflicts with attention backend {selected_name!r}."
        )

    from .attention import Gemma4FAAttentionBackend

    logger.info_once(
        "Gemma4 FA fast path enabled for full attention D512: "
        "Q>16 or Q=5 when supported, with Triton fallback."
    )
    return Gemma4FAAttentionBackend
