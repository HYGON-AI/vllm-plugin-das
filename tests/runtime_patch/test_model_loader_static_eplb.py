import importlib
from types import ModuleType, SimpleNamespace

import pytest

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map


def adapter():
    name = "vllm_hcu.patch.worker.framework_opt.patch_model_loader_static_eplb"
    assert importlib.util.find_spec(name), "static preload adapter missing"
    return importlib.import_module(name)


def test_initialize_binds_current_model_before_return(tmp_path, monkeypatch):
    api = adapter()
    config, _ = config_and_map(tmp_path)
    model = GenericMoE()
    module = ModuleType(api.TARGET_MODULE)
    def initialize_model(vllm_config, *, prefix="", model_class=None, model_config=None):
        assert vllm_config is config
        return model
    module.initialize_model = initialize_model
    # No already-imported alias may silently bypass the new boundary.
    for name in api.INITIALIZER_ALIASES:
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    api.apply_to_module(module)
    loaded = module.initialize_model(config)
    assert loaded._vllm_hcu_static_eplb_plan.layer_map(1) == (2, 0, 1, 2)
    assert loaded.moe_layers[1].routed_experts._vllm_hcu_static_eplb_row == (2, 0, 1, 2)
    assert api.apply_to_module(module) is False


def test_initialize_rejects_stale_alias_before_patch(monkeypatch):
    api = adapter()
    module = ModuleType(api.TARGET_MODULE)
    def initialize_model(vllm_config, *, prefix="", model_class=None, model_config=None):
        return None
    module.initialize_model = initialize_model
    alias = ModuleType(api.INITIALIZER_ALIASES[0])
    alias.initialize_model = lambda: None
    monkeypatch.setitem(__import__("sys").modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="alias"):
        api.apply_to_module(module)
    assert module.initialize_model is initialize_model


@pytest.mark.parametrize("format_name", ["dummy", "sharded_state", "tensorizer"])
def test_loader_gate_rejects_static_formats_that_bypass_rows(tmp_path, format_name):
    api = adapter()
    config, _ = config_and_map(tmp_path)
    with pytest.raises(ValueError, match="loader|format"):
        api.validate_static_loader(config, SimpleNamespace(load_format=format_name))


def test_loader_gate_preserves_default_and_inactive_formats(tmp_path):
    api = adapter()
    config, _ = config_and_map(tmp_path)
    api.validate_static_loader(config, SimpleNamespace(load_format="safetensors"))
    config.parallel_config._vllm_hcu_expert_map_path = None
    api.validate_static_loader(config, SimpleNamespace(load_format="dummy"))


def test_worker_inventory_arms_static_boundaries():
    from vllm_hcu.patch.worker import _FRAMEWORK_CALLBACKS
    modules = {spec.adapter for spec in _FRAMEWORK_CALLBACKS}
    assert "vllm_hcu.patch.worker.framework_opt.patch_model_loader_static_eplb" in modules
    assert "vllm_hcu.patch.worker.framework_opt.patch_model_loader_static_eplb_gate" in modules
    assert "vllm_hcu.patch.worker.framework_opt.patch_offline_eplb" in modules
