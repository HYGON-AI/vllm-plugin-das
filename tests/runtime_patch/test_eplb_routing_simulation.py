import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("balance,want", [(1.0, [8, 8, 8, 8]), (0.5, [16, 6, 5, 5]), (0.25, [32, 0, 0, 0])])
def test_routing_simulator_targets_requested_rank_balancedness(monkeypatch, balance, want):
    import vllm.model_executor.layers.fused_moe.router.routing_simulator_router as target
    name = "vllm_hcu.patch.worker.framework_opt.patch_routing_simulator"
    assert importlib.util.find_spec(name), "EPLB routing simulation adapter missing"
    api = importlib.import_module(name)
    monkeypatch.setattr(target.RoutingSimulator, "_routing_strategies", dict(target.RoutingSimulator._routing_strategies))
    monkeypatch.setattr(target, api._MARKER, False, raising=False)
    monkeypatch.setattr(api, "get_ep_world_size", lambda: 4)
    monkeypatch.setenv("VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS", str(balance))
    api.apply_to_module(target)
    hidden = torch.zeros(16, 4)
    logits = torch.zeros(16, 16)
    weights, ids = target.RoutingSimulator.simulate_routing(hidden, logits, api.STRATEGY_NAME, 2, torch.int32)
    assert torch.bincount((ids // 4).flatten(), minlength=4).tolist() == want
    assert (ids[:, 0] != ids[:, 1]).all()
    assert ids.dtype == torch.int32
    assert weights.tolist() == [[1., 1.]] * 16


@pytest.mark.parametrize("balance", ["nan", "1.1", "0.1"])
def test_invalid_simulated_balance_rejected(monkeypatch, balance):
    import vllm.model_executor.layers.fused_moe.router.routing_simulator_router as target
    name = "vllm_hcu.patch.worker.framework_opt.patch_routing_simulator"
    assert importlib.util.find_spec(name), "EPLB routing simulation adapter missing"
    api = importlib.import_module(name)
    monkeypatch.setattr(target.RoutingSimulator, "_routing_strategies", dict(target.RoutingSimulator._routing_strategies))
    monkeypatch.setattr(target, api._MARKER, False, raising=False)
    monkeypatch.setattr(api, "get_ep_world_size", lambda: 4)
    monkeypatch.setenv("VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS", balance)
    api.apply_to_module(target)
    with pytest.raises(ValueError):
        target.RoutingSimulator.simulate_routing(torch.zeros(16, 4), torch.zeros(16, 16), api.STRATEGY_NAME, 2)
