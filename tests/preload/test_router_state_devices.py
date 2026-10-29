import torch
import pytest
from types import SimpleNamespace

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


@pytest.mark.parametrize("load_format, expected_calls", [("auto", 0), ("ipc_cache", 2)])
def test_worker_only_reconciles_preloaded_models(monkeypatch, load_format, expected_calls):
    import vllm_hcu.patch.worker as worker_patches
    import vllm_hcu.runtime_compat.weight_cache as cache
    from vllm_hcu.v1.worker import HcuGPUWorker, Worker

    calls = []
    target, draft = object(), object()
    runner = SimpleNamespace(model=target, get_draft_model=lambda: draft)
    worker = object.__new__(HcuGPUWorker)
    worker.vllm_config = SimpleNamespace(load_config=SimpleNamespace(load_format=load_format))
    worker.model_runner = runner
    worker.device = torch.device("cpu")
    monkeypatch.setattr(Worker, "load_model", lambda self, **kwargs: None)
    monkeypatch.setattr(worker_patches, "validate_worker_patches", lambda **kwargs: None)
    monkeypatch.setattr(cache, "reconcile_router_state_devices",
                        lambda model, device: calls.append((model, device)) or ())
    worker.load_model()
    assert calls == [(target, worker.device), (draft, worker.device)][:expected_calls]


@pytest.mark.parametrize("load_format", ["auto", "ipc_cache"])
def test_worker_checks_config_for_all_loaders(monkeypatch, load_format):
    from vllm import envs
    import vllm_hcu.v1.config_consistency as consistency
    import vllm_hcu.v1.worker as worker_module

    class StopAfterCheck(Exception):
        pass

    worker = object.__new__(worker_module.HcuGPUWorker)
    worker.device_config = SimpleNamespace(device_type="cuda")
    worker.parallel_config = SimpleNamespace(distributed_executor_backend="external_launcher")
    worker.vllm_config = SimpleNamespace(load_config=SimpleNamespace(load_format=load_format))
    worker.model_config = SimpleNamespace(dtype=torch.float16, seed=0)
    worker.local_rank, worker.rank = 0, 0
    worker.distributed_init_method = "unused"
    worker.use_v2_model_runner = False
    calls = []
    monkeypatch.setattr(envs, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP", False)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda *args: None)
    monkeypatch.setattr(worker_module.current_platform, "check_if_supports_dtype", lambda *args: None)
    monkeypatch.setattr(worker_module, "init_worker_distributed_environment", lambda *args: None)
    monkeypatch.setattr(consistency, "validate_distributed_engine_config",
                        lambda *args: calls.append(args))
    def stop(*args):
        raise StopAfterCheck
    monkeypatch.setattr(worker_module, "set_random_seed", stop)
    with pytest.raises(StopAfterCheck):
        worker.init_device()
    assert len(calls) == 1
