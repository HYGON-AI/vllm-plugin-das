# SPDX-License-Identifier: Apache-2.0
"""DSpark group-derived buffer geometry and admission contracts."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from hcu_vllm_das import dspark_graph, dspark_ll
from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager


def config(width):
    return NS(
        model_config=NS(hf_config=NS(model_type="deepseek_v41")),
        speculative_config=NS(method="dspark"),
        parallel_config=NS(
            tensor_parallel_size=1, prefill_context_parallel_size=1,
            pipeline_parallel_size=1, decode_context_parallel_size=1,
            data_parallel_size=width, enable_expert_parallel=True,
            all2all_backend="deepep_low_latency"),
    )


@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_graph_policy_uses_dp_membership(width):
    cfg = config(width)
    assert dspark_graph.native_decode_dspark(cfg)
    cfg.speculative_config.target_parallel_config = cfg.parallel_config
    cfg.parallel_config = NS()
    assert dspark_graph.native_decode_dspark(cfg)


@pytest.mark.parametrize("field,value", [
    ("data_parallel_size", 0), ("data_parallel_size", 1),
    ("data_parallel_size", True), ("tensor_parallel_size", 2),
    ("prefill_context_parallel_size", 2), ("decode_context_parallel_size", 2),
    ("enable_expert_parallel", False), ("all2all_backend", "allgather_reducescatter"),
])
def test_graph_policy_rejects_other_execution_contracts(field, value):
    cfg = config(4)
    setattr(cfg.parallel_config, field, value)
    assert not dspark_graph.native_decode_dspark(cfg)


def manager(width):
    return Mock(spec=DeepEPLLAll2AllManager, world_size=width,
                dp_world_size=width, internode=False, support_fault_tolerance=False)


@pytest.mark.parametrize("width", [2, 4, 8])
def test_target_and_draft_share_one_native_buffer(width, monkeypatch):
    import deep_ep
    probes = []
    for name in ("is_ll_layered_opt_enabled", "is_ll_layered_develop_enabled"):
        monkeypatch.setattr(deep_ep.Buffer, name, lambda n: probes.append(n) or False)
    native = manager(width)
    shared = dspark_ll.SharedIPCManager(native, 256)
    handles = []
    for experts in (384, 128, 384):
        handles.append(shared.get_handle(dict(
            max_num_tokens_per_dp_rank=256, token_hidden_size=5120,
            num_ep_ranks=width, num_global_experts=experts,
            num_local_experts=experts // width)))
    assert probes == [width, width]
    assert handles[0] is handles[1] is handles[2]
    assert all(call.args[0] == dict(
        max_num_tokens_per_dp_rank=256, token_hidden_size=5120,
        num_ep_ranks=width, num_global_experts=384,
        num_local_experts=384 // width) for call in native.get_handle.call_args_list)
    with pytest.raises(ValueError, match="geometry"):
        shared.get_handle(dict(max_num_tokens_per_dp_rank=256, token_hidden_size=5120,
                               num_ep_ranks=width * 2, num_global_experts=128,
                               num_local_experts=128 // width))


@pytest.mark.parametrize("field,value,error", [
    ("world_size", 3, "matching"), ("dp_world_size", 8, "matching"),
    ("internode", True, "matching"), ("support_fault_tolerance", True, "matching"),
])
def test_shared_buffer_keeps_ownership_checks(field, value, error):
    native = manager(4)
    setattr(native, field, value)
    with pytest.raises(ValueError, match=error):
        dspark_ll.SharedIPCManager(native, 256)


def test_unshardable_experts_rejected():
    with pytest.raises(ValueError, match="evenly sharded"):
        dspark_ll.SharedIPCManager(manager(3), 256)
