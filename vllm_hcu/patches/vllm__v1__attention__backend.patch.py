# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.attention.backend.py
"""

PATCHES = [
(
"""
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
""",
"""
        from vllm_hcu.v1.attention.backends.fa_utils import hcu_ops
        torch.ops.hcu_ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
        )
""",
),

(
"""
@dataclass
class CommonAttentionMetadata:
""",
"""
@dataclass
class CpCommonAttentionMetadata:
    # sp related metadata
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    _seq_lens_cpu: torch.Tensor
    num_actual_tokens: int

    # lightly_cp 模式下表示 all-gather 后的全量 KV token 数
    num_kv_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    num_reqs: int

    block_table_tensor: torch.Tensor

    slot_mapping: torch.Tensor
    _num_computed_tokens_cpu: torch.Tensor

    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None

    def batch_size(self) -> int:
        return self.seq_lens.shape[0]
    
    @property
    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu


@dataclass
class CommonAttentionMetadata:
"""
),

(
"""
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
""",
"""
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    # lightly_cp 模式下表示 all-gather 后的全量 KV token 数
    num_kv_actual_tokens: int | None = None
    
    seq_indexes_list: list[int] | None = None
    scatter_indexes_tensor: torch.Tensor | None = None
    gather_indexes_tensor: torch.Tensor | None = None
    cp_common_metadata: CpCommonAttentionMetadata | None = None
"""
),

(
"""
            num_actual_tokens=num_actual_tokens,
""",
"""
            num_actual_tokens=num_actual_tokens,
            num_kv_actual_tokens=num_actual_tokens,
"""
),
]
