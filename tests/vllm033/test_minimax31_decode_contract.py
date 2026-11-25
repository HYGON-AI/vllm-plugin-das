"""Device-free refusal tests for decode publications and role readiness."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from hcu_vllm_das.parallel.minimax31_decode_contract import (
    partition, require_partition, ready_roles, validate_parallel)


def test_target_and_dense_draft_have_distinct_ownership():
    target = partition("target", 3, "GPU-3", "a"*64)
    draft = partition("draft", 3, "GPU-3", "b"*64)
    assert (target.dp_rank, target.ep_rank, target.expert_begin, target.expert_end) == (3, 3, 48, 64)
    assert (draft.dp_rank, draft.ep_size, draft.physical_experts) == (3, 1, 0)
    require_partition(target, target.to_dict())
    for changed in (replace(target, role="draft"), replace(target, dp_rank=4),
                    replace(target, gpu_uuid="other"), replace(target, pcp_size=8),
                    replace(target, weight_identity="c"*64)):
        with pytest.raises(ValueError): require_partition(target, changed.to_dict())
    bad = target.to_dict(); bad["dp_rank"] = True
    with pytest.raises(ValueError): require_partition(target, bad)


def test_both_roles_must_be_ready_on_the_same_eight_distinct_devices():
    rows = [dict(role=role, dp_rank=r, gpu_uuid="GPU-%d" % r)
            for role in ("target", "draft") for r in range(8)]
    assert ready_roles(rows, with_draft=True)
    for bad in (rows[:-1], rows + rows[:1], [dict(x, gpu_uuid="other") if
                x["role"] == "draft" and x["dp_rank"] == 2 else x for x in rows]):
        with pytest.raises(ValueError): ready_roles(bad, with_draft=True)


def test_pcp_cache_and_non_dp8_are_not_decode_publications():
    config = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
        data_parallel_size=8, prefill_context_parallel_size=1, decode_context_parallel_size=1,
        enable_eplb=False, expert_placement_strategy="linear", nnodes=1)
    assert validate_parallel(config)
    config.prefill_context_parallel_size = 8
    with pytest.raises(ValueError): validate_parallel(config)
    with pytest.raises(ValueError): partition("target", True, "GPU", "a"*64)
