# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CUDA graphs of the decoder replay layers on a step's replay batch.

Port of upstream ``vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py``
(#59532) for HCU / vLLM 0.29.1rc0.dev283.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import replace

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

from .decoder_replay import DecoderReplayLayers

try:
    from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
except ImportError:  # pragma: no cover - missing on 0.29.1rc0.dev283
    NATIVE_MEGA_MOE_BACKENDS = ()

# 0.29 may omit NATIVE_MEGA_MOE_BACKENDS while still shipping MegaMoE; the
# patched DeepseekV4Model.forward casts input_ids to int64 for this backend.
_MEGA_MOE_INPUT_ID_BACKENDS = frozenset(NATIVE_MEGA_MOE_BACKENDS) | {
    "deep_gemm_mega_moe",
}


class DecoderReplayCudaGraphManager(CudaGraphManager):
    """Breakable PIECEWISE graphs of the replay layers keyed by the replay batch's
    padded size. They read the layer inputs from static buffers."""

    def __init__(self, vllm_config: VllmConfig, device, layers: DecoderReplayLayers):
        compilation = copy(vllm_config.compilation_config)
        scheduler = vllm_config.scheduler_config
        sizes = list(
            getattr(compilation, "decoder_replay_cudagraph_capture_sizes", None) or []
        )
        if not sizes:
            bound = min(
                compilation.max_cudagraph_capture_size,
                scheduler.max_num_seqs * layers.window,
            )
            # Replay batches hold at least `window` rows: step by window, or by 1/8
            # of the power-of-two interval once wider (256 in [2048, 4096)).
            sizes, s = [bound], layers.window
            while s < bound:
                sizes.append(s)
                s += max(layers.window, 1 << max(s.bit_length() - 4, 0))
        sizes = sorted(set(sizes))
        if max(sizes) > scheduler.max_num_batched_tokens:
            raise ValueError(
                "decoder_replay_cudagraph_capture_sizes must not exceed "
                "max_num_batched_tokens"
            )
        compilation.cudagraph_capture_sizes = sizes
        compilation.max_cudagraph_capture_size = max(sizes)
        replay_config = copy(vllm_config)
        replay_config.compilation_config = compilation
        super().__init__(replay_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.layers = layers
        self.breakable_cg_runner: BreakableCUDAGraphWrapper = BreakableCUDAGraphWrapper(
            layers.run_layers, replay_config
        )
        config, act = vllm_config.model_config.hf_config, vllm_config.model_config.dtype
        assert isinstance(act, torch.dtype)
        hc, h, f32 = config.hc_mult, config.hidden_size, torch.float32
        # DeepseekV4Model.forward casts input_ids to int64 for native MegaMoE only.
        kernel_config = getattr(vllm_config, "kernel_config", None)
        moe_backend = getattr(kernel_config, "moe_backend", None)
        ids = (
            torch.int64
            if moe_backend in _MEGA_MOE_INPUT_ID_BACKENDS
            else torch.int32
        )
        # hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
        shapes = [(h,), (), (), (hc,), (hc, 1), (hc, hc), (hc, h)]
        dtypes = [act, torch.int64, ids, f32, f32, f32, act]
        n = compilation.max_cudagraph_capture_size
        self.input_buffers = [
            torch.zeros(n, *s, dtype=d, device=device) for s, d in zip(shapes, dtypes)
        ]

    def capture_replay_graphs(
        self, prepare: Callable[[BatchExecutionDescriptor], None]
    ) -> None:
        """Capture each size on the replay batch ``prepare`` sets on the layers."""

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            prepare(desc)
            assert self.layers.replay_batch is not None
            context = self.layers.replay_batch.forward_context
            inputs = [buf[: desc.num_tokens] for buf in self.input_buffers]

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                with override_forward_context(
                    replace(context, cudagraph_runtime_mode=cg_mode)
                ):
                    self.breakable_cg_runner(*inputs)

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay CUDA graphs")
        self.layers.replay_batch = None

    def run(self, rows: torch.Tensor, inputs: tuple) -> tuple[torch.Tensor, ...]:
        """Replay the forward context's graph on ``rows`` of ``inputs``."""
        num_rows = rows.shape[0]
        for buf, t in zip(self.input_buffers, inputs):
            torch.index_select(t, 0, rows, out=buf[:num_rows])
        batch_descriptor = get_forward_context().batch_descriptor
        assert batch_descriptor is not None
        num_tokens = batch_descriptor.num_tokens
        outputs = self.breakable_cg_runner(
            *(buf[:num_tokens] for buf in self.input_buffers)
        )
        return tuple(out[:num_rows] for out in outputs)


__all__ = ["DecoderReplayCudaGraphManager"]
