# SPDX-License-Identifier: Apache-2.0
"""Verify optional EPLB patches do not mutate feature-off processes."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_state import PatchRegistry
from vllm_hcu.patch import worker


@pytest.mark.parametrize("loaded_first", [False, True])
def test_deferred_callback_preserves_original_until_enabled(monkeypatch, loaded_first):
    name = "hcu_eplb_isolation_target"
    target = ModuleType(name)
    original = lambda: "official"
    target.forward = original
    calls = []
    def install(module):
        calls.append(module)
        module.forward = lambda: "eplb"
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    try:
        if loaded_first:
            monkeypatch.setitem(sys.modules, name, target)
        coordinator.register_callback(
            "test.eplb.gated", name, install, feature_enabled=False,
            defer_until_enabled=True,
        )
        monkeypatch.setitem(sys.modules, name, target)
        coordinator.drain_ready_callbacks()
        assert target.forward is original
        assert calls == []
        assert coordinator.registrations()[0].status == "armed"
        coordinator.set_feature_enabled("test.eplb.gated", True)
        assert target.forward() == "eplb"
        coordinator.drain_ready_callbacks()
        coordinator.set_feature_enabled("test.eplb.gated", True)
        assert calls == [target]
    finally:
        coordinator.reset_for_tests()


def test_existing_feature_off_callbacks_keep_previous_semantics(monkeypatch):
    name = "hcu_eplb_legacy_target"
    target = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, target)
    calls = []
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    try:
        coordinator.register_callback("test.legacy", name, calls.append,
                                      feature_enabled=False)
        assert calls == [target]
    finally:
        coordinator.reset_for_tests()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mapping", [False, True])
def test_model_adapter_follows_native_eplb_flag(enabled, mapping):
    parallel = {"enable_eplb": enabled}
    config = ({"parallel_config": parallel} if mapping else
              SimpleNamespace(parallel_config=SimpleNamespace(**parallel)))
    states = worker._feature_states(config, worker.HcuFeatureConfig())
    assert states["eplb"] is enabled


def test_all_new_eplb_adapters_defer_installation():
    expected = {
        "patch_deepseek_v41_eplb": "eplb",
        "patch_model_loader_static_eplb": "offline_eplb",
        "patch_static_expert_mapping": "offline_eplb",
        "patch_offline_eplb": "eplb_control",
        "patch_eplb_communicator": "offline_eplb",
        "patch_static_eplb_router": "offline_eplb",
    }
    specs = {spec.adapter.rsplit(".", 1)[-1]: spec
             for group in worker._ALL_CALLBACK_GROUPS for spec in group}
    for name, feature in expected.items():
        spec = specs[name]
        assert spec.feature == feature
        assert spec.defer_until_enabled


def test_base_router_patch_does_not_replace_mapping_method():
    from vllm_hcu.patch.worker.op_opt.moe import patch_base_router
    def official(topk_ids, expert_load_view, logical_to_physical_map,
                 logical_replica_count, record_enabled, num_unpadded_tokens=None):
        return topk_ids
    class BaseRouter:
        def _apply_eplb_mapping(self, topk_ids):
            return topk_ids
    original = BaseRouter._apply_eplb_mapping
    target = ModuleType(patch_base_router.TARGET_MODULE)
    target.BaseRouter = BaseRouter
    target.dbo_current_ubatch_id = lambda: 0
    target.eplb_map_to_physical_and_record = official
    patch_base_router.apply_to_module(target)
    assert BaseRouter._apply_eplb_mapping is original


_GATED_SPECS = [
    spec for group in worker._ALL_CALLBACK_GROUPS for spec in group
    if spec.defer_until_enabled
]


@pytest.mark.parametrize("spec", _GATED_SPECS, ids=lambda spec: spec.adapter.rsplit(".", 1)[-1])
def test_worker_registration_defers_eplb_installers(monkeypatch, spec):
    # Exercise dispatcher wiring, including a target loaded before Worker config.
    target = ModuleType("hcu_eplb_worker_registration_target")
    original = lambda: "official"
    target.forward = original
    monkeypatch.setitem(sys.modules, target.__name__, target)
    adapter = ModuleType(spec.adapter)
    adapter.PATCH_ID = "test.worker.gated_eplb"
    adapter.TARGET_MODULE = target.__name__
    adapter.TARGETS = (target.__name__ + ".forward",)
    calls = []
    def apply_to_module(module):
        if calls:
            return False
        calls.append(module)
        module.forward = lambda: "eplb"
        return True
    adapter.apply_to_module = apply_to_module
    monkeypatch.setattr(worker, "_ALL_CALLBACK_GROUPS", ((spec,),))
    monkeypatch.setattr(worker, "_load_adapter", lambda name: adapter)
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    try:
        worker._register_callbacks(coordinator)
        coordinator.drain_ready_callbacks()
        assert target.forward is original
        assert calls == []
        registration = coordinator.registrations()[0]
        assert registration.status == "armed"
        coordinator.set_feature_enabled(registration.patch_id, True)
        assert target.forward() == "eplb"
        assert calls == [target]
        worker._register_callbacks(coordinator)
        assert calls == [target]
    finally:
        coordinator.reset_for_tests()
