# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Warm Kimi K3's cold KDA BF16 projection before checkpoint loading."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.kimi_k3.amd.kda"
PATCH_ID = "worker.op_opt.kimi_k3.kda_bf16_gemm_warmup"
TARGETS = (f"{TARGET_MODULE}.KimiK3DeltaAttention.__init__",)
_MARKER = "_vllm_hcu_kimi_k3_kda_gemm_warmup_applied"
_WRAPPER = "_vllm_hcu_kimi_k3_kda_gemm_warmup_wrapper"
_PROFILE_M = 6288
_warmup_done = False


def apply_to_module(module: ModuleType) -> bool:
    global _warmup_done

    kda = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        kda, "KimiK3DeltaAttention", f"{TARGET_MODULE}.KimiK3DeltaAttention"
    )
    wrapped = ((cls, "__init__", TARGETS[0], _WRAPPER),)
    if already_applied(kda, _MARKER, wrapped):
        return False

    original_init = require_callable(cls, "__init__", TARGETS[0])

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        global _warmup_done

        original_init(self, *args, **kwargs)
        if _warmup_done:
            return

        linear = getattr(self, "in_proj_qkvgfab", None)
        weight = getattr(linear, "weight", None)
        if weight is None or weight.device.type != "cuda":
            # Model construction can be inspected on CPU. The actual HCU worker
            # must hit the device branch before loading its checkpoint.
            return
        if weight.dtype != torch.bfloat16:
            return

        hidden_size = int(weight.shape[-1])
        logger = getattr(kda, "logger", None)
        if logger is not None:
            logger.info(
                "Running Kimi K3 KDA BF16 projection warmup before weight load: "
                "M=%d, packed_output=%d, K=%d, device=%s",
                _PROFILE_M,
                int(weight.shape[0]),
                hidden_size,
                weight.device,
            )
        scratch = torch.zeros(
            (_PROFILE_M, hidden_size), device=weight.device, dtype=weight.dtype
        )
        with torch.inference_mode():
            self.in_proj_qkvgfab(scratch)
            torch.cuda.synchronize(weight.device)
        del scratch
        _warmup_done = True
        if logger is not None:
            logger.info("Kimi K3 KDA BF16 projection warmup completed")

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "__init__", hcu_init)
    setattr(kda, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
