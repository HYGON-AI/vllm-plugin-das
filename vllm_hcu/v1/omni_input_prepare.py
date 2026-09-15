# SPDX-License-Identifier: Apache-2.0
"""Opt-in input preparation for the Qwen3-Omni thinker on HCU.

Keep the upstream runner and its input-transfer fence. The M-RoPE buffer has
the token-major storage introduced by commit 1ad6f388, while exposing the
existing [3, T] interface to Omni's position calculation and fixup methods.
"""
from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.utils import CpuGpuBuffer

from vllm_hcu.platforms import envs as henvs

logger = init_logger("vllm.hcu.omni_input_prepare")


class TokenMajorMropeBuffer(CpuGpuBuffer):
    """A flat, pinned token-major buffer with compatible [3, T] views.

    Both CPU and GPU views have strides (1, 3). Their [:, :n] windows are
    dense and have identical strides, so PyTorch can copy the contiguous
    3*n values directly without channel-major packing or a temporary tensor.
    The address and strides remain stable across CUDA graph replays.
    """

    def __init__(self, num_tokens: int, *, device: torch.device, pin_memory: bool):
        super().__init__(3 * num_tokens, dtype=torch.int64,
                         device=device, pin_memory=pin_memory)
        self.flat_cpu = self.cpu
        self.flat_gpu = self.gpu
        self.cpu = self.flat_cpu.view(num_tokens, 3).T
        self.gpu = self.flat_gpu.view(num_tokens, 3).T
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        values = self.flat_cpu.numel() if n is None else 3 * n
        self.flat_gpu[:values].copy_(self.flat_cpu[:values], non_blocking=True)
        return self.gpu if n is None else self.gpu[:, :n]

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        values = self.flat_cpu.numel() if n is None else 3 * n
        self.flat_cpu[:values].copy_(self.flat_gpu[:values], non_blocking=True)
        return self.cpu if n is None else self.cpu[:, :n]



def initialize_omni_input_prepare(runner: Any) -> None:
    mrope = henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_1D_MROPE
    config = runner.model_config
    # Limit the opt-in integration to the architecture and stage validated
    # here; other Omni models retain their own input/payload semantics.
    supported = (
        config.model_arch == "Qwen3OmniMoeForConditionalGeneration"
        and config.model_stage == "thinker"
        and not config.async_chunk
        and not config.is_encoder_decoder
        and runner.speculative_config is None
    )
    if not supported:
        if not mrope:
            return None
        raise ValueError("HCU Omni input preparation requires a non-speculative "
                         "Qwen3-Omni thinker without async_chunk")
    # vLLM's serialized AOT graphs do not re-check input strides when loaded.
    # The layout must therefore participate in VllmConfig.compute_hash before
    # load_model constructs any compiled submodel. Mark BOTH switch states:
    # an off run must not load a graph created by an older, unkeyed on run.
    additional = runner.vllm_config.additional_config
    if additional is not None and not isinstance(additional, dict):
        raise ValueError("HCU Omni input preparation requires dict additional_config")
    runner.vllm_config.additional_config = {
        **(additional or {}),
        "hcu_omni_mrope_layout": "token_major_v1" if mrope and runner.uses_mrope else "channel_major_v1",
    }
    logger.info("HCU Omni M-RoPE cache layout: %s",
                runner.vllm_config.additional_config["hcu_omni_mrope_layout"])
    if not mrope:
        return None
    if mrope and runner.uses_mrope:
        runner.mrope_positions = TokenMajorMropeBuffer(
            runner.max_num_tokens + 1, device=runner.device,
            pin_memory=PIN_MEMORY)
    logger.info("HCU Omni input preparation: token_major_mrope=%s",
                bool(mrope and runner.uses_mrope))
