# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal window indexing over engine-managed sliding MLA pages."""

from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseBackend
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm_hcu.models.hy_v4_flash.runtime import is_hcu_platform


class FlashWindowIndexGroupBuilder(SparseMLAIndexGroupBuilder):
    """Keep window-owned KV out of HiSparse without patching shared Core.

    The window manager owns these pages. Passing HiSparse's config here would
    attach a host/hot cache handle which the window manager never binds. This
    is the model-scoped equivalent of upstream vLLM #69026's SWA exclusion.
    """

    def register_layer(self, is_index_producing_layer, vllm_config=None, **kwargs):
        return super().register_layer(is_index_producing_layer, None, **kwargs)


class FlashSparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The direct paged kernel never gathers evicted SWA history.
        self.fp8_use_mixed_batch = True

    def _build_fp8_mixed_decode_prefill(self):
        if not is_hcu_platform():
            return super()._build_fp8_mixed_decode_prefill()
        # Core's imported factory belongs to the NVIDIA extension. Pair the
        # HCU kernel with its own scheduler object without rebinding Core.
        from vllm_hcu.v1.attention.ops.flashmla import get_mla_metadata

        scheduler, _ = get_mla_metadata()
        return FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler,
            cache_lens=self.max_model_len_tensor[:1],
            dummy_block_table=self.dummy_block_table[:1],
        )


class FlashSparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_builder_cls():
        return FlashSparseMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSWABackend(FlashSparseBackend):
    @classmethod
    def supports_sliding_window(cls):
        return True


def window_indices(positions: torch.Tensor, window: int, width: int) -> torch.Tensor:
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover its causal window")
    columns = torch.arange(width, device=positions.device)
    indices = positions[:, None] - window + 1 + columns
    live = (positions[:, None] >= 0) & (columns < window) & (indices >= 0)
    return torch.where(live, indices, -1).int()


class FlashWindowIndexer(nn.Module):
    def __init__(self, window: int, topk_indices_buffer, attention_prefix: str):
        super().__init__()
        self.window = window
        self.topk_indices_buffer = topk_indices_buffer
        self.attention_prefix = attention_prefix
        if topk_indices_buffer.shape[1] < window:
            raise ValueError("SWA index width must cover its causal window")

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("Flash SWA requires per-layer metadata")
        meta: Any = metadata[self.attention_prefix]
        if meta.num_actual_tokens == 0 or meta.seq_lens.numel() == 0:
            return output.fill_(-1)
        rows = torch.arange(output.shape[0], device=output.device)
        starts = meta.query_start_loc
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = meta.seq_lens[requests] - (starts[requests + 1] - starts[requests])
        logical = logical + rows - starts[requests]
        live = (rows < meta.num_actual_tokens) & (rows < starts[-1])
        output.copy_(
            window_indices(torch.where(live, logical, -1), self.window, output.shape[1])
        )
        return output
