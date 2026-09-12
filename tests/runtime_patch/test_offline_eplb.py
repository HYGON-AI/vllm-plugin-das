import importlib
import json
from types import MethodType, SimpleNamespace

import pytest
import torch

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan


def adapter():
    name = "vllm_hcu.patch.worker.framework_opt.patch_offline_eplb"
    assert importlib.util.find_spec(name), "offline state adapter missing"
    return importlib.import_module(name)


def setup_state(tmp_path, monkeypatch, mode="static"):
    import vllm.distributed.eplb.eplb_state as upstream
    api = adapter()
    for cls, names in [(upstream.EplbState, ("add_model", "step", "rearrange")),
                       (upstream, ("_commit_eplb_maps", "rearrange_expert_weights_inplace"))]:
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(upstream, api._MARKER, False, raising=False)
    model = GenericMoE()
    config, path = config_and_map(tmp_path)
    parallel = config.parallel_config
    parallel.num_ubatches = 0
    parallel.eplb_config = SimpleNamespace(use_async=False, window_size=2,
        step_interval=10, policy="default", communicator="torch_gloo", log_balancedness=False)
    group = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=None,
        device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2), barrier=lambda: None)
    monkeypatch.setattr(upstream, "get_ep_group", lambda: group)
    monkeypatch.setattr(upstream, "get_eplb_group", lambda: group)
    monkeypatch.setattr(upstream, "get_node_count", lambda: 1)
    monkeypatch.setattr(upstream, "create_eplb_communicator", lambda **kwargs: object())
    def no_transfer(*args, **kwargs):
        pytest.fail("expert weights rearranged")
    monkeypatch.setattr(upstream, "rearrange_expert_weights_inplace", no_transfer)
    api.apply_to_module(upstream)
    for layer in model.moe_layers:
        layer.eplb_state = upstream.EplbLayerState()
        layer.get_expert_weights = lambda layer=layer: list(layer.routed_experts.parameters())
        layer.set_eplb_state = layer.eplb_state.set_layer_state
    if mode == "static":
        bind_static_eplb_plan(config, model)
    else:
        parallel._vllm_hcu_expert_map_path = None
    if mode == "record":
        parallel._vllm_hcu_expert_map_record_path = str(tmp_path / "record.json")
    state = upstream.EplbState(parallel, torch.device("cpu"))
    model_config = SimpleNamespace(model="test", compute_hash=lambda: "test")
    return api, upstream, state, model, model_config, path


def test_static_add_commits_two_maps_without_weight_rearrange(tmp_path, monkeypatch):
    _, upstream, state, model, config, _ = setup_state(tmp_path, monkeypatch)
    state.add_model(model, config)
    live = state.model_states["test"]
    assert live.physical_to_logical_map.tolist() == [[0, 1, 2, 1], [2, 0, 1, 2]]
    assert live.logical_replica_count.tolist() == [[1, 2, 1], [1, 1, 2]]
    assert model.moe_layers[1].eplb_state.logical_to_physical_map[2, :2].tolist() == [0, 3]
    assert not state.is_async


@pytest.mark.parametrize("kwargs", [{}, {"is_dummy": True}, {"is_profile": True}, {"log_stats": True}])
def test_static_step_is_noop(tmp_path, monkeypatch, kwargs):
    _, _, state, model, config, _ = setup_state(tmp_path, monkeypatch)
    state.add_model(model, config)
    before = (state.expert_rearrangement_step, state.expert_load_window_step)
    assert state.step(**kwargs) is None
    assert before == (state.expert_rearrangement_step, state.expert_load_window_step)
    assert state.rearrange(is_profile=True) is None


def test_runtime_changed_file_rejected_before_state_publication(tmp_path, monkeypatch):
    _, _, state, model, config, path = setup_state(tmp_path, monkeypatch)
    path.write_text(json.dumps({"model_maps": {"GenericMoE": {
        "physical_to_logical_map": [[2, 0, 1, 2], [0, 1, 2, 1]]}}}))
    with pytest.raises(ValueError, match="changed|mismatch"):
        state.add_model(model, config)
    assert not state.model_states


def test_static_state_requires_bound_plan(tmp_path, monkeypatch):
    _, _, state, model, config, _ = setup_state(tmp_path, monkeypatch)
    del model._vllm_hcu_static_eplb_plan
    with pytest.raises(ValueError, match="bound|plan"):
        state.add_model(model, config)
    assert not state.model_states


def test_dynamic_add_uses_current_initial_map(tmp_path, monkeypatch):
    _, _, state, model, config, _ = setup_state(tmp_path, monkeypatch, "dynamic")
    state.add_model(model, config)
    assert state.model_states["test"].physical_to_logical_map.tolist() == [[0, 1, 2, 0]] * 2


def test_record_mode_records_initial_map_without_changing_it(tmp_path, monkeypatch):
    _, _, state, model, config, _ = setup_state(tmp_path, monkeypatch, "record")
    state.add_model(model, config)
    payload = json.loads((tmp_path / "record.json").read_text())
    assert payload["model_maps"]["GenericMoE"]["physical_to_logical_map"] == [[0, 1, 2, 0]] * 2
    assert state.model_states["test"].physical_to_logical_map.tolist() == [[0, 1, 2, 0]] * 2


def test_corrupt_record_file_is_not_overwritten(tmp_path):
    api = adapter()
    path = tmp_path / "record.json"
    path.write_text("corrupt")
    with pytest.raises(ValueError):
        api.record_offline_expert_map(path, model_key="GenericMoE", model_name="test",
            model_class="GenericMoE", physical_to_logical_map=torch.tensor([[0, 1, 2, 0]]),
            num_logical_experts=3, num_redundant_experts=1)
    assert path.read_text() == "corrupt"


def test_sidecar_normalizes_eplb_fields():
    from vllm_hcu.patch.config import HcuFeatureConfig
    config = HcuFeatureConfig.from_mapping({"expert_map_path": "/tmp/map.json",
        "eplb_static_dispatch_policy": "locality_fair"})
    assert config.expert_map_path == "/tmp/map.json"
    assert config.to_dict()["eplb_static_dispatch_policy"] == "locality_fair"


@pytest.mark.parametrize("payload", [dict(expert_map_path="a", expert_map_record_path="b"),
    dict(expert_map_path=True), dict(eplb_disable_rearrange=1),
    dict(eplb_static_dispatch_policy="unknown")])
def test_invalid_sidecar_fields_fail_closed(payload):
    from vllm_hcu.patch.config import HcuFeatureConfig
    with pytest.raises((ValueError, TypeError)):
        HcuFeatureConfig.from_mapping(payload)


def test_engine_args_strips_eplb_fields_to_sidecar():
    import inspect
    from vllm_hcu.patch.platform.core_fix.patch_engine_args import _normalise_constructor_kwargs
    def constructor(self, *, eplb_config=None, additional_config=None):
        pass
    kwargs = {"eplb_config": {"expert_map_path": "/tmp/map.json", "num_redundant_experts": 1}}
    feature = _normalise_constructor_kwargs(inspect.signature(constructor), object(), (), kwargs)
    assert kwargs["eplb_config"] == {"num_redundant_experts": 1}
    assert feature.expert_map_path == "/tmp/map.json"


def test_real_engine_args_cli_preserves_official_validation_and_sidecar(monkeypatch):
    import vllm.engine.arg_utils as module
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm_hcu.patch.config import get_hcu_config
    from vllm_hcu.patch.platform.core_fix import patch_engine_args

    for owner in (module.EngineArgs, module.AsyncEngineArgs):
        for name in ("__init__", "from_cli_args", "add_cli_args", "create_engine_config",
                     "_vllm_hcu_original_init", "_vllm_hcu_original_from_cli_args",
                     "_vllm_hcu_original_add_cli_args", "_vllm_hcu_original_create_engine_config"):
            if name in vars(owner):
                monkeypatch.setattr(owner, name, vars(owner)[name])
            else:
                monkeypatch.setattr(owner, name, None, raising=False)
    monkeypatch.setattr(module, patch_engine_args._MARKER,
                        getattr(module, patch_engine_args._MARKER, False), raising=False)
    patch_engine_args.apply_to_module(module)
    parser = module.EngineArgs.add_cli_args(FlexibleArgumentParser())
    raw = dict(expert_map_path="/tmp/map.json", static_dispatch_policy="locality_fair",
               num_redundant_experts=1)
    args = module.EngineArgs.from_cli_args(parser.parse_args(["--eplb-config", json.dumps(raw)]))
    assert args.eplb_config.num_redundant_experts == 1
    assert get_hcu_config(args).expert_map_path == "/tmp/map.json"
    assert get_hcu_config(args).eplb_static_dispatch_policy == "locality_fair"
    with pytest.raises(SystemExit):
        parser.parse_args(["--eplb-config", json.dumps(raw | {"unknown_field": True})])


def test_record_candidate_never_moves_weights_or_commits_live_map(tmp_path, monkeypatch):
    _, upstream, state, model, config, _ = setup_state(tmp_path, monkeypatch, "record")
    state.add_model(model, config)
    candidate = torch.tensor([[2, 0, 1, 2], [0, 1, 2, 1]])
    state.policy = SimpleNamespace(rebalance_experts=lambda *args: candidate)
    state._allreduce_list = lambda values: values
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0))
    state.rearrange()
    payload = json.loads((tmp_path / "record.json").read_text())
    assert payload["model_maps"]["GenericMoE"]["physical_to_logical_map"] == candidate.tolist()
    assert state.model_states["test"].physical_to_logical_map.tolist() == [[0, 1, 2, 0]] * 2


def test_disable_rearrange_keeps_load_collection(tmp_path, monkeypatch):
    _, _, state, model, config, _ = setup_state(tmp_path, monkeypatch, "dynamic")
    state.parallel_config._vllm_hcu_eplb_disable_rearrange = True
    state.add_model(model, config)
    state.model_states["test"].expert_load_pass.fill_(2)
    state._allreduce_list = lambda values: values
    state.policy = SimpleNamespace(rebalance_experts=lambda *args: torch.tensor([[0, 1, 2, 0]] * 2))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0))
    assert state.rearrange() is None
    assert state.model_states["test"].expert_load_pass.sum() == 16


def test_static_state_applies_locality_order_to_current_maps(tmp_path, monkeypatch):
    _, upstream, state, model, config, _ = setup_state(tmp_path, monkeypatch)
    state.parallel_config._vllm_hcu_eplb_static_dispatch_policy = "locality_fair"
    group = SimpleNamespace(world_size=2, rank_in_group=1,
        device_group=SimpleNamespace(rank=lambda: 1, size=lambda: 2))
    monkeypatch.setattr(upstream, "get_ep_group", lambda: group)
    state.add_model(model, config)
    assert state.model_states["test"].logical_to_physical_map[0, 1, :2].tolist() == [3, 1]
    assert model.moe_layers[0].eplb_state.logical_to_physical_map[1, :2].tolist() == [3, 1]


def test_gloo_communicator_needs_no_device_profile_reservation(monkeypatch):
    import vllm.distributed.eplb.eplb_communicator as module
    name = "vllm_hcu.patch.worker.framework_opt.patch_eplb_communicator"
    assert importlib.util.find_spec(name), "Gloo profile policy adapter missing"
    api = importlib.import_module(name)
    monkeypatch.setattr(module, api._MARKER, False, raising=False)
    cls = module.TorchDistGlooStagedEplbCommunicator
    monkeypatch.setattr(cls, "needs_profile_buffer_reservation", cls.needs_profile_buffer_reservation, raising=False)
    # Restore the inherited shape before the adapter audits it.
    del cls.needs_profile_buffer_reservation
    api.apply_to_module(module)
    assert not object.__new__(cls).needs_profile_buffer_reservation
    assert object.__new__(module.TorchDistNcclEplbCommunicator).needs_profile_buffer_reservation


@pytest.mark.parametrize("as_dict", [False, True])
def test_worker_eplb_sidecar_survives_object_and_dict_transport(as_dict):
    from vllm_hcu.patch import config as module
    assert hasattr(module, "bind_hcu_eplb_config"), "worker EPLB sidecar binding missing"
    values = {"additional_config": {"hcu": {"expert_map_path": "/tmp/map.json",
              "eplb_static_dispatch_policy": "locality_fair"}}, "parallel_config": {}}
    config = values if as_dict else SimpleNamespace(**(values | {"parallel_config": SimpleNamespace()}))
    module.bind_hcu_eplb_config(config)
    parallel = config["parallel_config"] if as_dict else vars(config.parallel_config)
    assert parallel["_vllm_hcu_expert_map_path"] == "/tmp/map.json"
    assert parallel["_vllm_hcu_eplb_static_dispatch_policy"] == "locality_fair"
