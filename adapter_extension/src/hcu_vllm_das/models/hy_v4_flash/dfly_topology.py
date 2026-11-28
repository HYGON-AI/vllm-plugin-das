# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 target EP ownership and independent dense DFly DP replicas.

Target DP coordination and expert collectives remain Core runner / CT owned.
The dense TP1 draft has no EP/DP collective and uses only its local target KV.
"""

from copy import copy


def validate_standard_hy4_parallel(parallel, *, enforce_eager):
    for field in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"Standard DFly requires {field}=1")
    dp = parallel.data_parallel_size
    if (
        type(dp) is not int
        or dp != 16
        or parallel.tensor_parallel_size != 1
        or not parallel.enable_expert_parallel
        or not 0 <= parallel.data_parallel_rank < dp
        or parallel.all2all_backend
        not in ("deepep_high_throughput", "deepep_low_latency")
        or not enforce_eager
        or any(
            getattr(parallel, name, False)
            for name in ("enable_dbo", "enable_eplb", "enable_elastic_ep")
        )
    ):
        raise ValueError(
            "Standard HY4 VL/DFly requires eager TP1/DP16/EP16 with fixed DeepEP HT/LL and no DBO/EPLB"
        )


def validate_standard_dfly_parallel(parallel, *, enforce_eager):
    return validate_standard_hy4_parallel(parallel, enforce_eager=enforce_eager)


def make_rank_local_draft_parallel(target):
    draft = copy(target)
    draft.enable_expert_parallel = False
    draft.is_moe_model = False
    return draft


def validate_rank_local_dfly_step(dp_size, dp_rank, state, *, local_reqs, dummy_run):
    if dp_size == 1 or dummy_run:
        return
    if (
        dp_size != 16
        or state is None
        or not state.eager
        or len(state.num_tokens_across_dp) != dp_size
        or not 0 <= dp_rank < dp_size
        or not 0 <= local_reqs <= state.num_reqs
    ):
        raise ValueError(
            "DP16 DFly requires the target runner's existing eager DP synchronization"
        )


def validate_standard_dfly_runtime_parallel(actual, target, *, enforce_eager):
    validate_standard_dfly_parallel(target, enforce_eager=enforce_eager)
    for field in (
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(actual, field) != getattr(target, field):
            raise ValueError(f"DFly live parallel owner differs from target: {field}")
    if getattr(actual, "is_moe_model", None) is False:
        if actual.enable_expert_parallel:
            raise ValueError("DFly dense draft cannot participate in target EP")
    else:
        validate_standard_dfly_parallel(actual, enforce_eager=enforce_eager)
