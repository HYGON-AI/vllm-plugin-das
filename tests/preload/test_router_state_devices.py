import torch

from vllm_hcu.runtime_compat.weight_cache import reconcile_router_state_devices


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e_score_correction_bias = torch.nn.Parameter(torch.ones(4))
        self.plain_router_bias = torch.ones(4)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()
        # This mirrors implementations that keep the bias as an unregistered
        # tensor attribute rather than a Parameter/buffer.
        self.router.router_bias = torch.ones(4)


def test_reconcile_moves_registered_and_plain_router_state():
    model = _Model()
    moved = reconcile_router_state_devices(model, torch.device("meta"))

    assert "router.e_score_correction_bias" in moved
    assert "router.router_bias" in moved
    assert model.router.e_score_correction_bias.device.type == "meta"
    assert model.router.router_bias.device.type == "meta"
    assert isinstance(model.router.e_score_correction_bias, torch.nn.Parameter)


def test_reconcile_is_idempotent_on_target_device():
    model = _Model()
    reconcile_router_state_devices(model, torch.device("meta"))
    assert reconcile_router_state_devices(model, torch.device("meta")) == ()
