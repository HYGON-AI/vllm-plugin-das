# SPDX-License-Identifier: Apache-2.0
"""Bounded capacity slack for dynamically padded HT MoE scratch buffers."""
from math import prod
import os

import torch


def get_moe_workspaces(manager, shapes_and_dtypes, growth_bytes: int):
    """Return exact views while rounding the underlying shared capacity.

    Use the manager's public simultaneous allocation API so its alignment,
    locking, stream and ubatch lifetime rules remain authoritative. A trailing
    byte view reserves slack; it never changes offsets of the real buffers.
    Per slot, additional requested capacity is less than one growth bucket.
    No tensor or manager is cached by this helper.
    """
    if growth_bytes < 0 or growth_bytes % 256:
        raise ValueError("MoE workspace growth must be nonnegative and 256-byte aligned")
    if growth_bytes == 0:
        return manager.get_simultaneous(*shapes_and_dtypes)
    total_bytes = sum(
        ((prod(shape) * dtype.itemsize + 255) // 256) * 256
        for shape, dtype in shapes_and_dtypes
    )
    slack = (-total_bytes) % growth_bytes
    if slack == 0:
        return manager.get_simultaneous(*shapes_and_dtypes)
    views = manager.get_simultaneous(
        *shapes_and_dtypes, ((slack,), torch.uint8)
    )
    return views[:-1]


def step5_decode_workspace_plan(config, dtype):
    """Bound the simultaneous native MoE buffers for TP4/DP4 Step5 decode.

    The speculative acceptance policy does not affect these buffer shapes.
    Other parallel layouts retain the ordinary workspace planner.
    """
    from vllm_hcu.runtime_compat.step5 import is_step5_config
    if not is_step5_config(config):
        return None
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    spec = config.speculative_config
    model = config.model_config.hf_text_config
    from vllm_hcu.runtime_compat.step5 import is_step5_tp4_deepep_ll
    if (
        is_step5_tp4_deepep_ll(config)
        and scheduler.max_num_batched_tokens == 512
        and scheduler.max_num_seqs == 64
        and dtype == torch.bfloat16
        and (model.hidden_size, model.moe_intermediate_size,
             model.moe_num_experts, model.moe_top_k) == (4096, 1536, 352, 8)
        and spec is not None and spec.method == "mtp" and spec.num_speculative_tokens == 3
    ):
        # Reserve the reviewed masked-expert workspace before transport starts.
        # It remains inside Worker's measured non-KV memory budget.
        return {"shapes": ((5 * 1024**3 // dtype.itemsize,),),
                "required_bytes": 5 * 1024**3}
    chunk = int(os.environ.get("EP_FUSED_MOE_CHUNK", "16384"))
    if not (
        parallel.tensor_parallel_size == parallel.data_parallel_size == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.enable_expert_parallel
        and not parallel.enable_dbo
        and parallel.all2all_backend == "allgather_reducescatter"
        and scheduler.max_num_batched_tokens == 16384
        and scheduler.max_num_seqs == 64
        and dtype == torch.bfloat16
        and model.hidden_size == 4096
        and model.moe_intermediate_size == 1536
        and model.moe_num_experts == 352
        and model.moe_top_k == 8
        and spec is not None
        and spec.method == "mtp"
        and spec.num_speculative_tokens == 3
        and chunk == 16384
    ):
        return None
    full_rows = scheduler.max_num_batched_tokens * parallel.data_parallel_size
    local_experts = model.moe_num_experts // (
        parallel.tensor_parallel_size * parallel.data_parallel_size
    )
    padded_rows = min(full_rows, chunk) * model.moe_top_k + local_experts * 256
    width = max(model.hidden_size, 2 * model.moe_intermediate_size)
    shapes = ((padded_rows, width), (padded_rows, width), (full_rows, model.hidden_size))
    return {
        "shapes": shapes,
        "required_bytes": sum(((prod(shape) * dtype.itemsize + 255) // 256) * 256 for shape in shapes),
    }


def reserve_step5_decode_workspace(runner):
    """Reserve scratch inside measured profiling before device collectives."""
    plan = step5_decode_workspace_plan(runner.vllm_config, runner.dtype)
    if plan is None:
        return
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.v1.worker.workspace import current_workspace_manager

    group = get_ep_group()
    manager = current_workspace_manager()
    if not (
        group.world_size == 16
        and str(dist.get_backend(group.cpu_group)) == "gloo"
        and manager._num_ubatches == manager._num_lanes == 1
        and not manager.is_locked()
        and len(manager._current_workspaces) == 1
        and manager._current_workspaces[0] is None
    ):
        raise RuntimeError("Step5 decode requires a fresh, single-slot EP16 workspace")
    dist.barrier(group=group.cpu_group)
    views = manager.get_simultaneous(*[(shape, runner.dtype) for shape in plan["shapes"]])
    del views
    dist.barrier(group=group.cpu_group)
