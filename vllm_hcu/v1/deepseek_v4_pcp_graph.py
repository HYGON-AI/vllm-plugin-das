# SPDX-License-Identifier: Apache-2.0
"""Decode-only graph dispatch for replicated DeepSeek V4 PCP batches."""

from copy import copy
from dataclasses import replace
from types import MethodType

import numpy as np

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def install_decode_dispatch(runner):
    """Keep every PCP prefill/mixed batch outside target CUDA graphs."""
    manager = runner.cudagraph_manager
    if getattr(manager, "_hcu_pcp_decode_dispatch", False):
        return
    align_draft_graph_buckets(runner)
    original = manager.dispatch

    def dispatch(self, num_reqs, num_tokens, uniform_token_count,
                 num_active_loras=0, max_query_len=None, num_ubatches=1):
        if getattr(runner, "_hcu_pcp_graph_has_prefill", False):
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens,
                num_reqs=num_reqs, num_active_loras=num_active_loras,
                num_ubatches=num_ubatches,
            )
        return original(
            num_reqs, num_tokens, uniform_token_count, num_active_loras,
            max_query_len, num_ubatches,
        )

    manager.dispatch = MethodType(dispatch, manager)
    manager._hcu_pcp_decode_dispatch = True


def prepare_decode_graph_metadata(manager, batch):
    """Pad metadata-only request flags and invalidate unused captured slots."""
    num_tokens = batch.num_tokens
    padded_tokens = batch.num_tokens_after_padding
    slots = manager._global_slot_mappings[:, :padded_tokens]
    slots[:, num_tokens:].fill_(manager._pad_slot_id)
    # Target graphs capture the canonical BlockTables slot buffer; the
    # replicated runtime metadata reads the PCP copy. Invalidate both tails.
    manager._block_tables.slot_mappings[:, num_tokens:padded_tokens].fill_(
        manager._pad_slot_id
    )
    missing = batch.num_reqs_after_padding - batch.is_prefilling_np.size
    if missing:
        batch = replace(
            batch,
            is_prefilling_np=np.pad(
                batch.is_prefilling_np, (0, missing), constant_values=False
            ),
        )
    return batch, slots


def align_draft_graph_buckets(runner):
    """Use the same request buckets despite different target/draft widths."""
    speculator = getattr(runner, "speculator", None)
    draft = getattr(speculator, "query_cudagraph_manager", None)
    if draft is None or draft.cudagraph_mode == CUDAGraphMode.NONE:
        return
    target = runner.cudagraph_manager.compilation_config
    width = runner.decode_query_len
    requests = sorted({
        (size + width - 1) // width
        for size in target.cudagraph_capture_sizes
        if ((size + width - 1) // width) * width <= target.max_cudagraph_capture_size
        and (size + width - 1) // width <= runner.max_num_reqs
    })
    if not requests:
        return
    config = copy(draft.vllm_config)
    compilation = config.compilation_config = copy(draft.compilation_config)
    compilation.cudagraph_capture_sizes = [
        count * speculator.num_query_per_req for count in requests
    ]
    compilation.max_cudagraph_capture_size = compilation.cudagraph_capture_sizes[-1]
    speculator.query_cudagraph_manager = type(draft)(
        config, runner.device, draft.cudagraph_mode,
        decode_query_len=speculator.num_query_per_req,
    )
