# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Qwen3.5 LoRA cudagraph compatibility for HCU."""

from __future__ import annotations

import os


def install_qwen35_lora_cudagraph_compat() -> None:
    """Disable unsafe piecewise cudagraphs for Qwen3.5 with LoRA."""

    from vllm.config import CUDAGraphMode
    from vllm.v1 import cudagraph_dispatcher as dispatcher_module

    dispatcher_cls = dispatcher_module.CudagraphDispatcher
    if getattr(
        dispatcher_cls, "_hcu_qwen35_lora_piecewise_cudagraph_patch_applied", False
    ):
        return

    original_dispatch = dispatcher_cls.dispatch
    original_get_capture_descs = dispatcher_cls.get_capture_descs

    def _should_patch_qwen35_lora(self) -> bool:
        if os.environ.get("VLLM_HCU_QWEN35_LORA_ALLOW_UNSAFE_COMPILE") == "1":
            return False
        model_config = getattr(self.vllm_config, "model_config", None)
        architecture = getattr(model_config, "architecture", None)
        return (
            architecture == "Qwen3_5ForConditionalGeneration"
            and getattr(self.vllm_config, "lora_config", None) is not None
        )

    def patched_dispatch(
        self,
        num_tokens,
        uniform_decode=False,
        has_lora=False,
        num_active_loras=0,
        valid_modes=None,
        invalid_modes=None,
    ):
        if _should_patch_qwen35_lora(self):
            invalid_modes = set() if invalid_modes is None else set(invalid_modes)
            invalid_modes.add(CUDAGraphMode.PIECEWISE)
        return original_dispatch(
            self,
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
        )

    def patched_get_capture_descs(self):
        capture_descs = original_get_capture_descs(self)
        if not _should_patch_qwen35_lora(self):
            return capture_descs

        filtered_descs = []
        for mode, descs in capture_descs:
            if mode == CUDAGraphMode.PIECEWISE:
                dispatcher_module.logger.warning(
                    "Qwen3.5 with LoRA on HCU skips PIECEWISE cudagraph "
                    "capture because the GDN LoRA path is unstable in "
                    "piecewise graph mode."
                )
                descs = []
            if descs:
                filtered_descs.append((mode, descs))
        return filtered_descs

    dispatcher_cls.dispatch = patched_dispatch
    dispatcher_cls.get_capture_descs = patched_get_capture_descs
    dispatcher_cls._hcu_qwen35_lora_piecewise_cudagraph_patch_applied = True
