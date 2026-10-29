# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU DeepSeek sparse-MLA indexer runtime adapter."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.mla.indexer"
PATCH_ID = "worker.op_opt.mla.indexer_hcu"
TARGETS = (
    f"{TARGET_MODULE}.split_indexer_prefill_chunks",
    f"{TARGET_MODULE}.split_decodes_and_prefills",
    f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder.build",
)
_MARKER = "_vllm_hcu_mla_indexer_applied"
_WRAPPER = "_vllm_hcu_mla_indexer_wrapper"


def _wrap_prefill_chunk_builder(build_prefill):
    @functools.wraps(build_prefill)
    def hcu_prefill_chunk(
        start_idx, end_idx, query_start_loc, query_start_loc_cpu,
        uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
        block_table, compress_ratio, query_slice=None, skip_kv_gather=False,
        dcp_rank=0, dcp_world_size=1, cp_kv_cache_interleave_size=1,
    ):
        result = build_prefill(
            start_idx, end_idx, query_start_loc, query_start_loc_cpu,
            uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
            block_table, compress_ratio, query_slice, skip_kv_gather,
            dcp_rank, dcp_world_size, cp_kv_cache_interleave_size,
        )
        from vllm_hcu.platforms import envs as henvs
        if (result is None or not henvs.VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING
                or dcp_world_size != 1 or compress_ratio != 1):
            return result
        from vllm import envs
        from vllm_hcu.v1.attention.ops.prefill_mqa import plan_prefill_mqa_chunks
        # Both tensors are already CPU metadata. Never pull CUDA row bounds
        # back to the host or run the planner in each model layer.
        if query_start_loc_cpu.device.type != "cpu" or compressed_seq_lens_cpu.device.type != "cpu":
            raise PatchCompatibilityError("prefill MQA grouping requires CPU request lengths")
        offsets = query_start_loc_cpu[start_idx:end_idx + 1].tolist()
        q_lens = [b - a for a, b in zip(offsets, offsets[1:])]
        k_lens = compressed_seq_lens_cpu[start_idx:end_idx].tolist()
        result.hcu_mqa_groups = plan_prefill_mqa_chunks(
            q_lens, k_lens,
            0 if query_slice is None else query_slice.start,
            sum(q_lens) if query_slice is None else query_slice.stop,
            min_saved_cells=henvs.VLLM_HCU_PREFILL_MQA_GROUP_MIN_SAVED_CELLS,
            max_logits_bytes=envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
        )
        return result
    setattr(hcu_prefill_chunk, _WRAPPER, True)
    return hcu_prefill_chunk


def apply_to_module(module: ModuleType) -> bool:
    indexer = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(indexer, "DeepseekV32IndexerMetadataBuilder", f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder")
    wrapped = (
        (indexer, "split_indexer_prefill_chunks", TARGETS[0], _WRAPPER),
        (indexer, "split_decodes_and_prefills", TARGETS[1], _WRAPPER),
        (builder_cls, "build", TARGETS[2], _WRAPPER),
    )
    chunk_builder = getattr(indexer, "build_prefill_chunk_metadata", None)
    if chunk_builder is not None:
        wrapped += ((indexer, "build_prefill_chunk_metadata",
                     f"{TARGET_MODULE}.build_prefill_chunk_metadata", _WRAPPER),)
    if already_applied(indexer, _MARKER, wrapped):
        return False
    if chunk_builder is not None:
        require_exact_signature(
            chunk_builder, f"{TARGET_MODULE}.build_prefill_chunk_metadata",
            positional=("start_idx", "end_idx", "query_start_loc", "query_start_loc_cpu",
                        "uncompressed_seq_lens", "compressed_seq_lens", "compressed_seq_lens_cpu",
                        "block_table", "compress_ratio", "query_slice", "skip_kv_gather",
                        "dcp_rank", "dcp_world_size", "cp_kv_cache_interleave_size"),
            defaults={"query_slice": None, "skip_kv_gather": False,
                      "dcp_rank": 0, "dcp_world_size": 1, "cp_kv_cache_interleave_size": 1},
        )
    split_chunks = require_callable(indexer, "split_indexer_prefill_chunks", TARGETS[0])
    require_exact_signature(
        split_chunks, TARGETS[0],
        positional=("seq_lens_cpu", "query_lens_cpu", "workspace_size", "max_logits_bytes", "request_offset"),
        defaults={"request_offset": 0},
    )
    split_batch = require_callable(indexer, "split_decodes_and_prefills", TARGETS[1])
    require_exact_signature(
        split_batch, TARGETS[1],
        positional=("common_attn_metadata", "decode_threshold", "require_uniform", "treat_short_extends_as_decodes"),
        defaults={"decode_threshold": 1, "require_uniform": False,
                  "treat_short_extends_as_decodes": True},
    )
    build = require_callable(builder_cls, "build", TARGETS[2])
    require_exact_signature(
        build, TARGETS[2],
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )

    @functools.wraps(split_chunks)
    def hcu_split_chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
                         max_logits_bytes, request_offset=0):
        chunks = split_chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
                              max_logits_bytes, request_offset)
        return [
            (req_slice, query_slice)
            for req_slice, query_slice in chunks
            if query_slice.stop > query_slice.start
        ]

    @functools.wraps(split_batch)
    def hcu_split_batch(common_attn_metadata, decode_threshold=1,
                        require_uniform=False,
                        treat_short_extends_as_decodes=None):
        if treat_short_extends_as_decodes is None:
            treat_short_extends_as_decodes = (
                getattr(common_attn_metadata, "is_prefilling", None) is None
            )
        return split_batch(
            common_attn_metadata, decode_threshold, require_uniform,
            treat_short_extends_as_decodes,
        )

    @functools.wraps(build)
    def hcu_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = build(self, common_prefix_len, common_attn_metadata, fast_build)
        from vllm_hcu.model_executor.layers.attention.pcp import (
            effective_pcp_world_size,
        )

        result.num_kv_actual_tokens = getattr(
            common_attn_metadata, "num_kv_actual_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            result.pcp_world_size = effective_pcp_world_size(
                int(getattr(common_attn_metadata, "pcp_world_size", 1))
            )
        else:
            parallel_config = getattr(vllm_config, "parallel_config", None)
            pcp_world_size = getattr(
                parallel_config,
                "prefill_context_parallel_size",
                None,
            )
            if pcp_world_size is None:
                raise PatchCompatibilityError(
                    "required vLLM 0.25.1 prefill_context_parallel_size "
                    "is missing from sparse indexer metadata builder"
                )
            result.pcp_world_size = effective_pcp_world_size(
                int(pcp_world_size)
            )
        # HCU's AITER, lightop, and torch paged-MQA paths do not consume the
        # upstream DeepGEMM schedule metadata. In particular, lightop builds
        # its schedule internally when called with ``schedule_meta=None``.
        # Precomputing it here rejects valid flattened MTP2 batch widths (for
        # example 1024 requests x 3 target tokens) before model execution.
        return result

    for function in (hcu_split_chunks, hcu_split_batch, hcu_build):
        setattr(function, _WRAPPER, True)
    setattr(indexer, "_vllm_hcu_original_split_indexer_prefill_chunks", split_chunks)
    setattr(indexer, "_vllm_hcu_original_split_decodes_and_prefills", split_batch)
    setattr(builder_cls, "_vllm_hcu_original_build", build)
    setattr(indexer, "split_indexer_prefill_chunks", hcu_split_chunks)
    setattr(indexer, "split_decodes_and_prefills", hcu_split_batch)
    setattr(builder_cls, "build", hcu_build)
    if chunk_builder is not None:
        setattr(indexer, "build_prefill_chunk_metadata", _wrap_prefill_chunk_builder(chunk_builder))
    setattr(indexer, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
