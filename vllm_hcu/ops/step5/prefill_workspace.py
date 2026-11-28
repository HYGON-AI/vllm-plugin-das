# SPDX-License-Identifier: Apache-2.0
"""Account for long-context producer scores before KV capacity is selected."""
from __future__ import annotations

import torch

from vllm_hcu.runtime_compat.step5 import is_step5_pd_config


class PrefillScoreWorkspace:
    """Shared by DSA layers after each selector has consumed its score matrix.

    Producers execute without CUDA Graphs. The opaque indexer consumes scores
    before HCU DBO hands off at MoE. A stream change also fences prior readers
    before another score kernel can overwrite this allocation.
    """

    def __init__(self, max_rows: int, max_regions: int, device):
        if max_rows <= 0 or max_regions <= 0:
            raise ValueError("Prefill score capacity must be positive")
        self.max_rows = int(max_rows)
        self.max_regions = int(max_regions)
        self.buffer = torch.empty(
            self.max_rows * self.max_regions, dtype=torch.float32, device=device
        )
        self._stream = None
        self._fence = None

    def get(self, rows: int, regions: int, *, device=None, zero=False):
        if not 0 <= rows <= self.max_rows or not 1 <= regions <= self.max_regions:
            raise ValueError("Prefill scores exceed their startup reservation")
        if device is not None and torch.device(device) != self.buffer.device:
            raise ValueError("Prefill score workspace belongs to another device")
        if self.buffer.is_cuda:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PD prefill score workspace cannot be captured")
            stream = torch.cuda.current_stream(self.buffer.device)
            if self._stream is not None and stream != self._stream:
                fence = torch.cuda.Event()
                fence.record(self._stream)
                stream.wait_event(fence)
                self._fence = fence
            self._stream = stream
        # Flattening first preserves the scorer's original contiguous row
        # stride even when this call uses fewer columns than the reservation.
        result = self.buffer[:rows * regions].view(rows, regions)
        if zero:
            result.zero_()
        return result


def create_pd_prefill_workspace(state, config):
    """Account for producer scores even at small batch sizes and in compiled mode."""
    if state is None or not is_step5_pd_config(config):
        return None
    if (config.kv_transfer_config.kv_role != "kv_producer"
            or config.scheduler_config.max_num_batched_tokens <= 0
            or state.num_kv_groups <= 0):
        return None
    mode = getattr(getattr(config, "compilation_config", None), "cudagraph_mode", None)
    if not config.model_config.enforce_eager and getattr(mode, "name", None) != "NONE":
        return None
    device = getattr(getattr(config, "device_config", None), "device", None)
    if device is None or torch.device(device).type != "cuda":
        return None
    regions = (config.model_config.max_model_len + state.geometry.region_size - 1) // state.geometry.region_size
    return PrefillScoreWorkspace(
        config.scheduler_config.max_num_batched_tokens * state.num_kv_groups,
        regions, device,
    )
