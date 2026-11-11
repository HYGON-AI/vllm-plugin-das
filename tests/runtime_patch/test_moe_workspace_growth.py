# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.v1.worker.workspace import WorkspaceManager
from vllm_hcu.model_executor.layers.fused_moe.workspace_growth import get_moe_workspaces


@pytest.fixture
def manager(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("release"))
    return WorkspaceManager(torch.device("cpu")), calls


def test_growth_reuses_capacity_and_preserves_exact_views(manager):
    m, calls = manager
    first = get_moe_workspaces(m, (((17,), torch.float32), ((33,), torch.bfloat16)), 1024)
    first[0].fill_(7)
    first[1].fill_(9)
    assert first[0].shape == (17,) and first[1].dtype == torch.bfloat16
    assert first[0].untyped_storage().data_ptr() == first[1].untyped_storage().data_ptr()
    assert first[0].storage_offset() == 0
    assert first[1].storage_offset() * first[1].element_size() == 256
    assert torch.all(first[0] == 7) and torch.all(first[1] == 9)
    ptr = first[0].untyped_storage().data_ptr()
    del first
    second = get_moe_workspaces(m, (((65,), torch.float32), ((33,), torch.bfloat16)), 1024)
    assert second[0].shape == (65,)
    assert second[0].untyped_storage().data_ptr() == ptr
    assert calls == ["release"]
    assert m._current_workspaces[0].numel() == 1024


def test_disabled_policy_keeps_exact_capacity(manager):
    m, calls = manager
    get_moe_workspaces(m, (((17,), torch.float32),), 0)
    assert m._current_workspaces[0].numel() == 256
    get_moe_workspaces(m, (((65,), torch.float32),), 0)
    assert m._current_workspaces[0].numel() == 512
    assert len(calls) == 2


def test_bucket_boundary_and_locked_manager(manager):
    m, calls = manager
    get_moe_workspaces(m, (((256,), torch.float32),), 1024)
    assert m._current_workspaces[0].numel() == 1024
    m.lock()
    get_moe_workspaces(m, (((128,), torch.float32),), 1024)
    with pytest.raises(AssertionError, match="locked"):
        get_moe_workspaces(m, (((257,), torch.float32),), 1024)
    assert len(calls) == 1


@pytest.mark.parametrize("growth", [-256, 1])
def test_invalid_growth_rejected(manager, growth):
    m, calls = manager
    with pytest.raises(ValueError):
        get_moe_workspaces(m, (((1,), torch.float32),), growth)
    assert not calls


@pytest.mark.parametrize("backend,dbo,standard,master,expected_bytes", [
    ("deepep_auto", False, True, True, 1024 * 1024),
    ("deepep_high_throughput", False, True, True, 1024 * 1024),
    ("deepep_auto", True, True, True, 512),
    ("deepep_auto", False, False, True, 512),
    ("allgather_reducescatter", False, True, True, 512),
    ("deepep_auto", False, True, False, 512),
])
def test_modular_allocation_policy_guards(
    manager, monkeypatch, backend, dbo, standard, master, expected_bytes
):
    from types import SimpleNamespace
    from vllm_hcu.model_executor.layers.fused_moe import modular_kernel as mk
    m, calls = manager
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: m)
    monkeypatch.setattr(mk, "dbo_enabled", lambda: dbo)
    monkeypatch.setattr(mk.henvs, "VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setattr(mk.henvs, "VLLM_HCU_MOE_WORKSPACE_GROWTH_MB", 1)
    impl = mk.FusedMoEKernelModularImpl.__new__(mk.FusedMoEKernelModularImpl)
    impl.moe_parallel_config = SimpleNamespace(all2all_backend=backend)
    impl.fused_experts = SimpleNamespace(
        workspace_dtype=lambda dtype: dtype,
        activation_format=lambda: (mk.FusedMoEActivationFormat.Standard if standard
                                  else mk.FusedMoEActivationFormat.BatchedExperts),
        workspace_shapes=lambda M, *args: ((M, 4), (M, 8), (M, 4)),
    )
    w13, w2, out = impl._allocate_buffers(
        torch.bfloat16, torch.device("cpu"), 4, 4, 8, 4, 2, 16, 2, None, None
    )
    assert w13.shape == out.shape == (4, 4)
    assert w2.shape == (4, 8)
    assert w13.data_ptr() == out.data_ptr()
    w13.fill_(3)
    w2.fill_(5)
    assert torch.all(out == 3) and torch.all(w2 == 5)
    assert m._current_workspaces[0].numel() == expected_bytes
    assert len(calls) == 1
