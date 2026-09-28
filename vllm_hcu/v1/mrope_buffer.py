# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Token-major storage for vLLM-compatible multi-axis rotary positions."""

from __future__ import annotations

import torch
from vllm.v1.utils import CpuGpuBuffer


class TokenMajorMropeBuffer(CpuGpuBuffer):
    """A flat, pinned token-major buffer with compatible [3, T] views.

    Both CPU and GPU views have strides (1, 3). Their [:, :n] windows are
    dense and have identical strides, so PyTorch can copy the contiguous
    3*n values directly without channel-major packing or a temporary tensor.
    The address and strides remain stable across CUDA graph replays.
    """

    def __init__(self, num_tokens: int, *, device: torch.device, pin_memory: bool):
        super().__init__(
            3 * num_tokens, dtype=torch.int64, device=device, pin_memory=pin_memory
        )
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
