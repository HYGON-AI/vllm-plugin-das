# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm.v1.attention.backend import (
    CommonAttentionMetadata as UpstreamCommonAttentionMetadata,
)

from vllm.v1.worker.gpu.attn_utils import (
    Any,
    AttentionGroup,
    KVCacheConfig,
    Mapping,
    ModelSpecificAttnMetadata,
    Sequence,
    torch,
)


@dataclass
class CommonAttentionMetadata(UpstreamCommonAttentionMetadata):
    """Host block table mirror scoped to Step5 metadata construction."""

    block_table_np: object | None = None

    def unpadded(self, num_actual_tokens, num_actual_reqs):
        result = super().unpadded(num_actual_tokens, num_actual_reqs)
        result.block_table_np = (
            self.block_table_np[:num_actual_reqs]
            if self.block_table_np is not None
            else None
        )
        return result


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    is_prefilling: torch.Tensor | None = None,
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
    ubatch_idx: int = 0,
    seq_lens_cpu: torch.Tensor | None = None,
    num_computed_tokens_cpu: torch.Tensor | None = None,
    block_tables_np: Sequence[Any] | None = None,
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]
    # Exact host mirrors (see InputBatch.seq_lens_cpu); never used for capture,
    # where the dummy batch has none.
    if for_cudagraph_capture:
        seq_lens_cpu = num_computed_tokens_cpu = block_tables_np = None
    if seq_lens_cpu is not None:
        seq_lens_cpu = seq_lens_cpu[:num_reqs]
    if num_computed_tokens_cpu is not None:
        num_computed_tokens_cpu = num_computed_tokens_cpu[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # Per-group causal for hybrid drafters (mixed SWA/full attention).
        group_causal = (
            causal if isinstance(causal, (bool, torch.Tensor)) else causal.get(i, True)
        )

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        # Model-specific metadata (e.g. Mamba hybrid) may supply its own
        # padding-aware is_prefilling, which takes precedence over the default.
        group_is_prefilling = common_attn_metadata_extra_kwargs.pop(
            "is_prefilling", is_prefilling
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=group_causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            is_prefilling=group_is_prefilling,
            mm_req_doc_ranges=mm_req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            block_table_np=(
                block_tables_np[i][:num_reqs] if block_tables_np is not None else None
            ),
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(ubatch_idx)
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges


def prepare_step5_attn(
    self,
    input_batch: InputBatch,
    cudagraph_mode: CUDAGraphMode,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    for_capture: bool = False,
    ubatch_idx: int = 0,
) -> dict[str, Any]:
    if cudagraph_mode == CUDAGraphMode.FULL:
        # Use padded sizes - padding is handled by model_runner.prepare_attn.
        num_reqs = input_batch.num_reqs_after_padding
        num_tokens = input_batch.num_tokens_after_padding
    else:
        # For piecewise cudagraphs and eager, use unpadded sizes.
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
    query_start_loc_cpu = torch.from_numpy(
        input_batch.query_start_loc_np[: num_reqs + 1]
    )
    query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
    max_query_len = input_batch.max_query_len
    if max_query_len is None:
        max_query_len = input_batch.num_scheduled_tokens.max().item()
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
    if for_capture:
        # Capture with worst-case max_seq_len so the graph is valid at any replay.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
    req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    if (
        self.supports_mm_inputs
        and self.encoder_cache is not None
        and self.model_config.is_mm_prefix_lm
    ):
        req_doc_ranges = compute_mm_prefix_ranges(
            req_ids=input_batch.req_ids,
            mm_features=self.encoder_cache.mm_features,
            sliding_window=self.model_config.get_sliding_window(),
        )
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=input_batch.seq_lens,
        max_seq_len=max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
        positions=input_batch.positions,
        is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
        mm_req_doc_ranges=req_doc_ranges,
        for_cudagraph_capture=for_capture,
        rswa_prefix_lens=input_batch.prompt_lens,
        ubatch_idx=ubatch_idx,
        seq_lens_cpu=getattr(input_batch, "seq_lens_cpu", None),
        num_computed_tokens_cpu=getattr(input_batch, "num_computed_tokens_cpu", None),
        block_tables_np=getattr(input_batch, "block_tables_np", None),
    )
    return attn_metadata
