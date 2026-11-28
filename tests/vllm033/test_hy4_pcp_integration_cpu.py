# SPDX-License-Identifier: Apache-2.0
"""Real parallel-config admission for the prefill-only PCP producer."""

import pytest


def producer():
    from vllm.config import ParallelConfig

    return ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=16,
        decode_context_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="deepep_high_throughput",
        distributed_executor_backend="mp",
    )


def test_pcp_producer_requires_explicit_parallel_admission():
    from hcu_vllm_das.models.hy_v4_flash.dfly_topology import (
        validate_standard_dfly_parallel,
    )

    target = producer()
    with pytest.raises(ValueError):
        validate_standard_dfly_parallel(target, enforce_eager=True)
    validate_standard_dfly_parallel(target, enforce_eager=True, allow_pcp_producer=True)


def test_producer_private_draft_has_independent_world_and_no_ep():
    from hcu_vllm_das.models.hy4_standard.pcp_draft import make_producer_draft_parallel
    from hcu_vllm_das.models.hy_v4_flash.dfly_topology import (
        validate_standard_dfly_runtime_parallel,
    )

    target = producer()
    draft = make_producer_draft_parallel(target)
    validate_standard_dfly_runtime_parallel(
        draft, target, enforce_eager=True, allow_pcp_producer=True
    )
    draft.enable_expert_parallel = True
    with pytest.raises(ValueError):
        validate_standard_dfly_runtime_parallel(
            draft, target, enforce_eager=True, allow_pcp_producer=True
        )


@pytest.mark.parametrize(
    "change",
    [
        {"all2all_backend": "deepep_low_latency"},
        {"prefill_context_parallel_size": 8},
        {"enable_dbo": True},
        {"enable_eplb": True},
    ],
)
def test_pcp_opt_in_does_not_admit_other_topologies(change):
    from hcu_vllm_das.models.hy_v4_flash.dfly_topology import (
        validate_standard_dfly_parallel,
    )

    target = producer()
    for key, value in change.items():
        setattr(target, key, value)
    with pytest.raises(ValueError):
        validate_standard_dfly_parallel(
            target, enforce_eager=True, allow_pcp_producer=True
        )


def test_explicit_activation_registers_owned_factory_with_master_disabled(monkeypatch):
    from hcu_adapter.bridges import vllm_model_runner
    from hcu_adapter.extensions import ExtensionContext
    from hcu_vllm_das import hy4_pcp_dfly_activation as activation
    from hcu_vllm_das.models.hy4_standard import pcp_runner

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    activation.activate(
        ExtensionContext(
            adapter_version="test",
            vllm_version="0.33.0",
            requested_capabilities=frozenset({"hy4-pcp-dfly"}),
        )
    )
    activation.check_health()
    entry = vllm_model_runner._factories[activation.OWNER]
    assert entry.matches is pcp_runner.matches and entry.create is pcp_runner.create
    vllm_model_runner.require_model_runner_factory(entry)
