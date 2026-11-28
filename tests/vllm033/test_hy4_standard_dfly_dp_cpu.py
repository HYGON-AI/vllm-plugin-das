"""Rank-local dense draft policy; no distributed/GPU qualification claim."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from hcu_vllm_das.models.hy_v4_flash.dfly_topology import (
    make_rank_local_draft_parallel,
    validate_rank_local_dfly_step,
    validate_standard_dfly_parallel,
)


def parallel(**kw):
    values = {
        "data_parallel_size": 16,
        "data_parallel_rank": 3,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "all2all_backend": "deepep_high_throughput",
        "enable_dbo": False,
        "enable_eplb": False,
        "enable_elastic_ep": False,
        "is_moe_model": True,
    }
    return NS(**(values | kw))


def test_dp16_target_and_dense_draft_have_distinct_ownership():
    target = parallel()
    before = deepcopy(vars(target))
    validate_standard_dfly_parallel(target, enforce_eager=True)
    draft = make_rank_local_draft_parallel(target)
    assert vars(target) == before
    assert draft.data_parallel_size == 16 and draft.data_parallel_rank == 3
    assert draft.tensor_parallel_size == 1
    assert draft.enable_expert_parallel is False and draft.is_moe_model is False


@pytest.mark.parametrize(
    "changes",
    [
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"enable_expert_parallel": False},
        {"all2all_backend": "allgather_reducescatter"},
        {"enable_dbo": True},
        {"enable_eplb": True},
        {"enable_elastic_ep": True},
        {"data_parallel_size": 8},
        {"data_parallel_rank": 16},
    ],
)
def test_dp16_rejects_unqualified_ownership(changes):
    with pytest.raises(ValueError):
        validate_standard_dfly_parallel(parallel(**changes), enforce_eager=True)


def test_dp16_requires_eager():
    with pytest.raises(ValueError):
        validate_standard_dfly_parallel(parallel(), enforce_eager=False)


@pytest.mark.parametrize("rank", [0, 3, 15])
def test_existing_target_dp_sync_covers_rank_local_query(rank):
    state = NS(eager=True, num_reqs=7, num_tokens_across_dp=[0] * 16, skip_drafts=False)
    # Ranks may have empty or unequal local query batches; no new collective.
    validate_rank_local_dfly_step(16, rank, state, local_reqs=rank % 7, dummy_run=False)


@pytest.mark.parametrize(
    "state",
    [
        None,
        NS(eager=False, num_reqs=7, num_tokens_across_dp=[0] * 16),
        NS(eager=True, num_reqs=7, num_tokens_across_dp=[0] * 8),
    ],
)
def test_real_dp16_steps_require_existing_eager_target_sync(state):
    with pytest.raises(ValueError):
        validate_rank_local_dfly_step(16, 0, state, local_reqs=0, dummy_run=False)


def test_profile_does_not_invent_a_target_dp_collective():
    validate_rank_local_dfly_step(16, 0, None, local_reqs=10, dummy_run=True)


@pytest.mark.parametrize("rank", [0, 3, 15])
def test_real_proposer_empty_rank_requires_sync_then_returns_without_query(rank):
    import __future__

    import ast

    import torch

    path = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/dfly_speculator.py"
    )
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlySpeculator"
    )
    propose = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose"
    )
    propose.decorator_list = []
    scope = {"__package__": "hcu_vllm_das.models.hy_v4_flash"}
    # Execute only the checked-in method without importing GPU-only modules.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[propose], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    owner = NS(
        max_num_reqs=8,
        num_query_per_req=6,
        max_num_tokens=48,
        dp_size=16,
        dp_rank=rank,
        draft_tokens=torch.zeros(8, 5, dtype=torch.int64),
    )
    state = NS(eager=True, num_reqs=3, num_tokens_across_dp=[0] * 16)
    args = [
        owner,
        NS(num_reqs=0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    result = scope["propose"](*args, dp_sync_state=state)
    assert result.shape == (0, 5)
    with pytest.raises(ValueError, match="target runner"):
        scope["propose"](*args, dp_sync_state=None)


def test_runtime_checks_live_parallel_owner_as_well_as_spec_target():
    from hcu_vllm_das.models.hy_v4_flash.dfly_topology import (
        validate_standard_dfly_runtime_parallel,
    )

    target = parallel()
    draft = make_rank_local_draft_parallel(target)
    validate_standard_dfly_runtime_parallel(target, target, enforce_eager=True)
    validate_standard_dfly_runtime_parallel(draft, target, enforce_eager=True)
    draft.data_parallel_size = 8
    with pytest.raises(ValueError, match="live"):
        validate_standard_dfly_runtime_parallel(draft, target, enforce_eager=True)
    draft = make_rank_local_draft_parallel(target)
    draft.enable_expert_parallel = True
    with pytest.raises(ValueError, match="draft"):
        validate_standard_dfly_runtime_parallel(draft, target, enforce_eager=True)
