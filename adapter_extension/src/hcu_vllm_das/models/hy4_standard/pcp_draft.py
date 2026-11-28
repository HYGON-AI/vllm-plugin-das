# SPDX-License-Identifier: Apache-2.0
"""Valid rank-local dense DFly construction and sealed producer K=0 requests."""

from vllm.config import ParallelConfig


def make_producer_draft_parallel(target):
    if (
        target.tensor_parallel_size,
        target.data_parallel_size,
        target.prefill_context_parallel_size,
        target.decode_context_parallel_size,
        target.pipeline_parallel_size,
    ) != (1, 1, 16, 1, 1):
        raise ValueError("HY4 producer draft requires an actual PCP16 target")
    draft = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        distributed_executor_backend=target.distributed_executor_backend,
        max_parallel_loading_workers=target.max_parallel_loading_workers,
        disable_custom_all_reduce=target.disable_custom_all_reduce,
        ray_workers_use_nsight=target.ray_workers_use_nsight,
        placement_group=target.placement_group,
    )
    draft.is_moe_model = False
    if draft.world_size != 1 or draft.world_size_across_dp != 1:
        raise RuntimeError("HY4 dense draft retained a stale distributed world size")
    return draft


def require_producer_zero_drafts(*, dummy_run, is_profile, num_speculative_tokens):
    if dummy_run:
        return
    if num_speculative_tokens != 0 or type(num_speculative_tokens) is not int:
        raise ValueError(
            "HY4 prefill-only producer must receive the scheduler's explicit K=0; PCP decode DFly is unsupported"
        )
