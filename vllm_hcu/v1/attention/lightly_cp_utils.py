
import numpy as np

import torch

from vllm.forward_context import get_forward_context
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
    CpCommonAttentionMetadata,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.utils.math_utils import round_up

def pad_for_mla_cp(num_scheduled_tokens: int) -> int:
    tp_size = get_tensor_model_parallel_world_size()
    return round_up(num_scheduled_tokens, tp_size)

def distribute_tokens_to_cp_ranks(
        total_q_len: int,
        q_lens_cpu: np.ndarray,
        kv_lens_cpu: np.ndarray,
        tp_rank: int,
        tp_size: int,
        enable_lightly_cplb: bool,
        device: torch.device
    ):
    q_lens = []
    seq_count = 0
    seq_indexes = []
    kv_lens = []

    local_scatter_indexes_tensor = None
    gather_indexes_tensor = None

    if enable_lightly_cplb:
        rank_tokens = 0
        rank_pad_tokens = 0
        accu_q_start = 0
        scatter_indexes: list[int] = []
        num_requests = len(q_lens_cpu)
        for i in range(num_requests):
            req_q_len = q_lens_cpu[i]
            req_pad_q_len = round_up(q_lens_cpu[i], 2 * tp_size)
            kv_len = kv_lens_cpu[i]

            chunk_q_len = req_pad_q_len // (2 * tp_size)

            q_1_start = tp_rank * chunk_q_len
            q_1_end = (tp_rank + 1) * chunk_q_len
            q_2_start = req_pad_q_len - (tp_rank + 1) * chunk_q_len
            q_2_end = req_pad_q_len - tp_rank * chunk_q_len

            q_len_1 = (
                chunk_q_len
                if q_1_end <= req_q_len
                else max(0, req_q_len - q_1_start)
            )
            q_len_2 = (
                chunk_q_len
                if q_2_end <= req_q_len
                else max(0, req_q_len - q_2_start)
            )

            kv_len_1 = kv_len - req_q_len + min(req_q_len, q_1_end)
            kv_len_2 = kv_len - req_q_len + min(req_q_len, q_2_end)

            scatter_index1 = range(
                accu_q_start + q_1_start, accu_q_start + q_1_start + q_len_1
            )

            scatter_index2 = range(
                accu_q_start + q_2_start, accu_q_start + q_2_start + q_len_2
            )
            accu_q_start += req_q_len

            if q_len_1 > 0:
                q_lens.append(q_len_1)
                kv_lens.append(kv_len_1)
                seq_indexes.append(i)
                scatter_indexes.extend(scatter_index1)
                seq_count += 1
                rank_tokens += q_len_1

            if q_len_2 > 0:
                q_lens.append(q_len_2)
                kv_lens.append(kv_len_2)
                seq_indexes.append(i)
                scatter_indexes.extend(scatter_index2)
                seq_count += 1
                rank_tokens += q_len_2

            rank_pad_tokens += chunk_q_len * 2

        if len(scatter_indexes) < rank_pad_tokens:
            scatter_indexes.extend([-1] * (rank_pad_tokens - len(scatter_indexes)))

        local_scatter_indexes_tensor = torch.tensor(
            scatter_indexes, dtype=torch.int64, device=device
        )
        global_scatter_indexes_tensor = tensor_model_parallel_all_gather(
            local_scatter_indexes_tensor.contiguous(), dim=0
        )
        non_neg_mask = global_scatter_indexes_tensor != -1
        non_neg_values = global_scatter_indexes_tensor[non_neg_mask]
        non_neg_positions = torch.where(non_neg_mask)[0]
        sorted_indices = torch.argsort(non_neg_values)
        gather_indexes_tensor = non_neg_positions[sorted_indices]
    else:
        tokens_per_rank = (total_q_len + tp_size - 1) // tp_size
        start_token = tp_rank * tokens_per_rank
        end_token = min((tp_rank + 1) * tokens_per_rank, total_q_len)
        
        current_seq = 0
        current_pos = 0
        rank_tokens = min(tokens_per_rank, end_token - start_token)
        while start_token < end_token and current_seq < len(q_lens_cpu):
            q_len = q_lens_cpu[current_seq]
            q_start = current_pos
            q_end = current_pos + q_len
            kv_len = kv_lens_cpu[current_seq]

            # Find overlap between this sequence and rank's token range
            overlap_start = max(start_token, q_start)
            overlap_end = min(end_token, q_end)

            if overlap_start < overlap_end:
                # This sequence contributes tokens to this rank
                token_count = overlap_end - overlap_start
                q_lens.append(token_count)
                start_token = overlap_end
                seq_count += 1
                seq_indexes.append(current_seq)

                if q_end <= end_token:
                    kv_lens.append(kv_len)
                else:
                    kv_lens.append(kv_len - (q_end - end_token))

            current_pos = q_end
            current_seq += 1

    return (
        rank_tokens,
        np.array(q_lens, dtype=np.int32),
        seq_count,
        np.array(kv_lens, dtype=np.int32),
        local_scatter_indexes_tensor,
        gather_indexes_tensor,
        seq_indexes,
    )

def prepare_cp_metadata(
        num_reqs_padded: int,
        max_query_len: int,
        max_seq_len: int,
        num_tokens: int,
        block_table_gid_0: torch.Tensor,
        slot_mapping_gid_0: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        num_computed_tokens_cpu: torch.Tensor,
        query_start_loc_buf: CpuGpuBuffer,
        seq_lens_buf: CpuGpuBuffer,
        enable_lightly_cplb: bool,
    ):
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    cp_common_metadata = CpCommonAttentionMetadata(
        query_start_loc=query_start_loc.clone(),
        query_start_loc_cpu=query_start_loc_cpu.clone(),
        seq_lens=seq_lens.clone(),
        _seq_lens_cpu=seq_lens_cpu.clone(),
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens,
        num_kv_actual_tokens=num_tokens,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        _num_computed_tokens_cpu=num_computed_tokens_cpu
    )

    query_start_loc_cpu = query_start_loc_cpu[: num_reqs_padded + 1]
    q_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    kv_lens_cpu = seq_lens_cpu
    total_q_len = num_tokens
    total_kv_len = num_tokens

    (
        total_q_len,
        q_lens_cpu,
        seq_count,
        kv_lens_cpu,
        scatter_indexes_tensor,
        gather_indexes_tensor,
        seq_indexes_list,
    ) = distribute_tokens_to_cp_ranks(
        total_q_len,
        q_lens_cpu,
        kv_lens_cpu,
        tp_rank,
        tp_size,
        enable_lightly_cplb,
        query_start_loc.device
    )

    num_reqs = seq_count

    cu_num_tokens = np.cumsum(q_lens_cpu)
    query_start_loc_buf.np[0] = 0
    query_start_loc_buf.np[1 : num_reqs + 1] = cu_num_tokens
    query_start_loc_buf.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
    query_start_loc_buf.copy_to_gpu()
    q_acc_lens = query_start_loc_buf.gpu[: num_reqs + 1]
    q_acc_lens_cpu = query_start_loc_buf.cpu[: num_reqs + 1]
    max_q_len = max(q_acc_lens_cpu)

    seq_lens_buf.np[:num_reqs] = kv_lens_cpu
    seq_lens_buf.np[num_reqs:].fill(0)
    seq_lens_buf.copy_to_gpu()
    kv_lens = seq_lens_buf.gpu[:num_reqs]
    kv_lens_cpu = seq_lens_buf.cpu[:num_reqs]
    max_kv_len = max(kv_lens_cpu)

    num_computed_tokens_cpu = kv_lens_cpu - q_acc_lens_cpu[1:]
    blk_table_tensor = block_table_gid_0[seq_indexes_list]

    cm_base = CommonAttentionMetadata(
        query_start_loc=q_acc_lens,
        query_start_loc_cpu=q_acc_lens_cpu,
        seq_lens=kv_lens,
        _seq_lens_cpu=kv_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=total_q_len,
        max_query_len=max_q_len,
        max_seq_len=max_kv_len,
        block_table_tensor=blk_table_tensor,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        num_kv_actual_tokens=num_tokens,
        seq_indexes_list=seq_indexes_list,
        cp_common_metadata=cp_common_metadata,
        scatter_indexes_tensor=scatter_indexes_tensor,
        gather_indexes_tensor=gather_indexes_tensor
    )
    return cm_base

def lightly_cp_inputs_splitting(hidden_states: torch.Tensor,
                                positions: torch.Tensor,
                                residual: torch.Tensor,
                                inputs_embeds: torch.Tensor,
                                tp_size: int,
                                tp_rank: int,
                                ):

    scatter_indexes_tensor = get_forward_context().scatter_indexes_tensor
    if scatter_indexes_tensor is None:
        hidden_states_per_rank = torch.chunk(hidden_states, chunks=tp_size, dim=0)
        hidden_states = hidden_states_per_rank[tp_rank].contiguous()

        if positions is not None:
            positions_per_rank = torch.chunk(positions, chunks=tp_size, dim=0)
            positions = positions_per_rank[tp_rank].contiguous()

        if residual is not None:
            residual_per_rank = torch.chunk(residual, chunks=tp_size, dim=0)
            residual = residual_per_rank[tp_rank].contiguous()

        if inputs_embeds is not None:
            inputs_embeds_per_rank = torch.chunk(inputs_embeds, chunks=tp_size, dim=0)
            inputs_embeds = inputs_embeds_per_rank[tp_rank].contiguous()
    else:
        scatter_indexes_tensor = torch.where(scatter_indexes_tensor == -1, 0, scatter_indexes_tensor)
        hidden_states = torch.index_select(hidden_states, 0, scatter_indexes_tensor)

        if positions is not None:
            positions = torch.index_select(positions, 0, scatter_indexes_tensor)

        if residual is not None:
            residual = torch.index_select(residual, 0, scatter_indexes_tensor)

        if inputs_embeds is not None:
            inputs_embeds = torch.index_select(inputs_embeds, 0, scatter_indexes_tensor)

    return hidden_states, positions, residual, inputs_embeds
