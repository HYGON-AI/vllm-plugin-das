# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.models.spec_decode.llm_base_proposer  __init__
"""

PATCHES = [
(
"""
        if current_platform.is_rocm():
""",
"""
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
        # if not current_platform.is_rocm():
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
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
""",
"""
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
            
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
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

(
'''                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )
''',
'''                    sh = getattr(layer, "shared_head", None)
                    if sh is None or not hasattr(sh, "head"):
                        continue

                    has_own_trained_weights = False

                    if self.enable_multi_layers_mtp:
                        if hasattr(sh.head, "weight") and hasattr(
                            target_language_model.lm_head, "weight"
                        ):
                            mtp_head_weight = sh.head.weight
                            target_head_weight = target_language_model.lm_head.weight
                            if isinstance(mtp_head_weight, torch.Tensor) and isinstance(
                                target_head_weight, torch.Tensor
                            ):
                                if not torch.isnan(mtp_head_weight).any() and not (
                                    torch.equal(
                                        mtp_head_weight.cpu(), target_head_weight.cpu()
                                    )
                                ):
                                    has_own_trained_weights = True

                    if has_own_trained_weights:
                        logger.info(
                            "MTP model has its own trained shared_head weights. "
                            "Keeping separate from target model lm_head."
                        )
                    else:
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )
''',
),

################ lightly cp###########################

(
"""
    def _determine_batch_execution_and_padding(
""",
"""
    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (
            self.compilation_config.pass_config.enable_sp
            or getattr(self.vllm_config.parallel_config, "enable_custom_sp", False)
        ) and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _determine_batch_execution_and_padding(
""",
),

(
"""
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
""",
"""
        num_tokens = self._pad_for_sequence_parallelism(num_tokens)
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
""",
),

]
