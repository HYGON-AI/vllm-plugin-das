# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.models.spec_decode.eagle  __init__
"""

PATCHES = [
(
"""
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)
""",
"""
        # if current_platform.is_rocm():
        #     from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        #         ROCMAiterMLASparseMetadata,
        #     )
        #     from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

        #     rocm_types = [
        #         TritonAttentionMetadata,
        #         RocmAttentionMetadata,
        #         ROCMAiterMLASparseMetadata,
        #     ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            # if find_spec(
            #     AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            # ):
            #     from vllm.v1.attention.backends.rocm_aiter_fa import (
            #         AiterFlashAttentionMetadata,
            #     )

            #     rocm_types.append(AiterFlashAttentionMetadata)

            # # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            # from vllm.model_executor.layers.attention.mla_attention import (
            #     MLACommonMetadata,
            # )

            # rocm_types.append(MLACommonMetadata)

            # # FlexAttention backend support
            # from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            # rocm_types.append(FlexAttentionMetadata)

            # self.allowed_attn_types = tuple(rocm_types)
""",
),

################ lightly cp###########################
(
"""
from vllm.distributed.parallel_state import get_pp_group
""",
"""        
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_rank
from vllm_hcu.v1.attention.lightly_cp_utils import pad_for_mla_cp, prepare_cp_metadata
from vllm.utils.math_utils import cdiv, round_up
""",
),

(
"""
from vllm.v1.attention.backend import CommonAttentionMetadata
""",
"""        
from vllm.v1.attention.backend import CommonAttentionMetadata, CpCommonAttentionMetadata
""",
),

(
"""
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
""",
"""        
        max_batch_size = vllm_config.scheduler_config.max_num_seqs if not vllm_config.parallel_config.enable_lightly_cplb else vllm_config.scheduler_config.max_num_seqs * 2
""",
),

(
"""
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(max_batch_size, 1)
""",
"""        
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(max_batch_size, 1)

        self.scatter_indexes_tensor = None
        self.gather_indexes_tensor = None

        self.enable_lightly_cp = vllm_config.parallel_config.enable_lightly_cp
        self.enable_lightly_cplb = self.enable_lightly_cp and vllm_config.parallel_config.enable_lightly_cplb
        if self.enable_lightly_cp:
            self.query_start_loc = CpuGpuBuffer(
                max_batch_size + 1,
                dtype=torch.int32,
                pin_memory=is_pin_memory_available(),
                device=device,
                with_numpy=True,
            )

            self.seq_lens = CpuGpuBuffer(
                max_batch_size,
                dtype=torch.int32,
                pin_memory=is_pin_memory_available(),
                device=device,
                with_numpy=True,
            )
""",
),

(
"""
        per_layer_attn_metadata: dict[str, object] = {}
""",
"""        
        per_layer_attn_metadata: dict[str, object] = {}

        enable_lightly_cp = self.enable_lightly_cp and num_tokens > self.runner.lightly_cp_threshold
        if enable_lightly_cp:
            actual_num_tokens = num_tokens
            num_tokens = pad_for_mla_cp(num_tokens)

            common_attn_metadata = prepare_cp_metadata(
                num_reqs_padded=common_attn_metadata.num_reqs,
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
                num_tokens=actual_num_tokens,
                block_table_gid_0=common_attn_metadata.block_table_tensor,
                slot_mapping_gid_0=common_attn_metadata.slot_mapping,
                query_start_loc=common_attn_metadata.query_start_loc,
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                seq_lens=common_attn_metadata.seq_lens,
                seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
                num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
                query_start_loc_buf=self.query_start_loc,
                seq_lens_buf=self.seq_lens,
                enable_lightly_cplb=self.enable_lightly_cplb
            )
            self.scatter_indexes_tensor = common_attn_metadata.scatter_indexes_tensor
            self.gather_indexes_tensor = common_attn_metadata.gather_indexes_tensor

"""
),

(
"""
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
""",
"""        
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
            scatter_indexes_tensor=self.scatter_indexes_tensor,
            gather_indexes_tensor=self.gather_indexes_tensor,
            enable_lightly_cp=self.enable_lightly_cp and num_tokens > self.runner.lightly_cp_threshold,
            enable_lightly_cplb=self.enable_lightly_cplb
""",
),

(
"""
        common_attn_metadata.num_actual_tokens = batch_size
""",
"""        
        if enable_lightly_cp:
            common_attn_metadata = common_attn_metadata.cp_common_metadata

        common_attn_metadata.num_actual_tokens = batch_size
""",
),

(
"""
        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )
""",
"""        
        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            num_kv_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )
""",
),
################ lightly cp###########################

]
