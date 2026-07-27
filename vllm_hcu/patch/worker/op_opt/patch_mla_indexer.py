# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU DeepSeek sparse-MLA indexer runtime adapter."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_class, require_exact_signature

TARGET_MODULE = "vllm.v1.attention.backends.mla.indexer"
PATCH_ID = "worker.op_opt.mla.indexer_hcu"
TARGETS = (
    f"{TARGET_MODULE}.split_indexer_prefill_chunks",
    f"{TARGET_MODULE}.split_decodes_and_prefills",
    f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder.build",
)
_MARKER = "_vllm_hcu_mla_indexer_applied"
_WRAPPER = "_vllm_hcu_mla_indexer_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    indexer = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(indexer, "DeepseekV32IndexerMetadataBuilder", f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder")
    wrapped = (
        (indexer, "split_indexer_prefill_chunks", TARGETS[0], _WRAPPER),
        (indexer, "split_decodes_and_prefills", TARGETS[1], _WRAPPER),
        (builder_cls, "build", TARGETS[2], _WRAPPER),
    )
    if already_applied(indexer, _MARKER, wrapped):
        return False
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
        result.num_kv_actual_tokens = getattr(
            common_attn_metadata, "num_kv_actual_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        if indexer.current_platform.is_rocm() and result.decode is not None:
            try:
                from lightop import gemmopt
            except ImportError as exc:
                raise RuntimeError("HCU sparse MLA indexer requires lightop.gemmopt") from exc
            seq_lens = result.decode.seq_lens.contiguous()
            result.decode.seq_lens = seq_lens
            metadata = gemmopt.get_paged_mqa_logits_metadata(
                seq_lens, self.kv_cache_spec.storage_block_size, self.num_sms
            )
            self.scheduler_metadata_buffer = metadata
            result.decode.schedule_metadata = metadata
        return result

    for function in (hcu_split_chunks, hcu_split_batch, hcu_build):
        setattr(function, _WRAPPER, True)
    setattr(indexer, "_vllm_hcu_original_split_indexer_prefill_chunks", split_chunks)
    setattr(indexer, "_vllm_hcu_original_split_decodes_and_prefills", split_batch)
    setattr(builder_cls, "_vllm_hcu_original_build", build)
    setattr(indexer, "split_indexer_prefill_chunks", hcu_split_chunks)
    setattr(indexer, "split_decodes_and_prefills", hcu_split_batch)
    setattr(builder_cls, "build", hcu_build)
    setattr(indexer, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
