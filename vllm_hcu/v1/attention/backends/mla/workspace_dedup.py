# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Rank-local BF16 sparse MLA workspace plans and execution."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm_hcu.v1.attention.backends.mla.prefill_workspace import plan_prefill_workspace
from vllm_hcu.v1.workspace_block_tables import HOST_TABLE_ATTRIBUTE


@dataclass
class WorkspaceChunk:
    descriptors: torch.Tensor
    num_tokens: int
    tokens_slice: slice


@dataclass
class WorkspaceMetadata:
    block_starts: torch.Tensor
    chunks: list[WorkspaceChunk]


def _to_device(values: np.ndarray, device: torch.device) -> torch.Tensor:
    host = torch.from_numpy(values)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


def build_workspace_metadata(builder, common, metadata):
    extra = metadata.fp8_extra_metadata
    if metadata.fp8_use_mixed_batch or extra is None:
        return None
    if extra.prefill is None or extra.num_prefill_tokens == 0:
        return None
    table = getattr(common, "sparse_mla_workspace_block_table_cpu", None)
    if table is None:
        table = getattr(common.block_table_tensor, HOST_TABLE_ATTRIBUTE, None)
    if table is None:
        return None
    begin = extra.num_decodes
    end = begin + extra.num_prefills
    lengths = common.seq_lens_cpu_upper_bound[begin:end].tolist()
    config = builder.vllm_config
    factor = getattr(
        config.attention_config, "sparse_mla_prefill_workspace_factor", None
    )
    capacity = config.model_config.max_model_len * (5 if factor is None else factor)
    block_size = builder.kv_cache_spec.block_size
    plans = plan_prefill_workspace(
        np.asarray(table)[begin:end], lengths, block_size, capacity
    )
    row_map = np.full(
        (end - begin, common.block_table_tensor.shape[1]), -1, dtype=np.int32
    )
    starts = common.query_start_loc_cpu
    chunks = []
    for plan in plans:
        row_map[plan.req_start : plan.req_end, : plan.row_block_starts.shape[1]] = (
            plan.row_block_starts
        )
        chunks.append(
            WorkspaceChunk(
                descriptors=_to_device(
                    np.column_stack(
                        (plan.physical_blocks, plan.valid_tokens, plan.workspace_starts)
                    ),
                    builder.device,
                ),
                num_tokens=plan.num_tokens,
                tokens_slice=slice(
                    int(starts[begin + plan.req_start]),
                    int(starts[begin + plan.req_end]),
                ),
            )
        )
    return WorkspaceMetadata(_to_device(row_map, builder.device), chunks)


def gather_workspace_pages(cache, workspace, descriptors):
    """Gather packed E4M3 cache pages using the plugin's compiled operator."""
    from vllm_hcu import hcu_ops  # noqa: F401

    torch.ops.hcu_ops.cp_gather_and_upconvert_fp8_kv_cache_pages(
        cache, workspace, descriptors
    )


def forward_workspace_dedup(impl, q, cache, indices, metadata):
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        reshape_attn_output_for_spec_decode,
        reshape_query_for_spec_decode,
    )
    from vllm_hcu.v1.attention.backends.mla.workspace_indices import (
        triton_convert_req_index_to_global_index,
    )

    extra = metadata.fp8_extra_metadata
    plan = metadata.hcu_workspace_dedup
    mapped, counts = triton_convert_req_index_to_global_index(
        metadata.req_id_per_token,
        metadata.block_table,
        indices,
        BLOCK_SIZE=metadata.block_size,
        NUM_TOPK_TOKENS=indices.shape[1],
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=extra.prefill.request_ids,
        prefill_workspace_starts=extra.prefill.workspace_starts,
        prefill_workspace_block_starts=plan.block_starts,
        return_valid_counts=True,
    )
    output = q.new_empty(
        (metadata.num_actual_tokens, impl.num_heads, impl.kv_lora_rank)
    )
    decode_tokens = extra.num_decode_tokens
    if decode_tokens:
        query = reshape_query_for_spec_decode(q[:decode_tokens], extra.num_decodes)
        selection = mapped[:decode_tokens].view(extra.num_decodes, query.shape[1], -1)
        decoded, _ = impl._fp8_flash_mla_kernel(
            q=query,
            kv_c_and_k_pe_cache=cache,
            topk_indices=selection,
            kernel_metadata=extra.decode.kernel_metadata,
        )
        output[:decode_tokens] = reshape_attn_output_for_spec_decode(decoded)
    for chunk in plan.chunks:
        if chunk.num_tokens > impl.prefill_bf16_workspace.shape[0]:
            raise RuntimeError("Deduplicated MLA chunk exceeds the allocated workspace")
        workspace = impl.prefill_bf16_workspace[: chunk.num_tokens]
        gather_workspace_pages(cache, workspace, chunk.descriptors)
        selection = chunk.tokens_slice
        if selection.start == selection.stop:
            continue
        output[selection] = impl._bf16_flash_mla_kernel(
            q[selection], workspace, mapped[selection], counts[selection]
        )
    return output
