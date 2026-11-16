# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from vllm_hcu.patch.worker.framework_opt import (
    patch_model_loader_static_eplb,
    patch_static_expert_mapping,
)


def _initializer_module(model):
    module = ModuleType(patch_model_loader_static_eplb.TARGET_MODULE)

    def initialize_model(
        vllm_config,
        *,
        prefix="",
        model_class=None,
        model_config=None,
    ):
        del vllm_config, prefix, model_class, model_config
        assert not hasattr(model, "_vllm_hcu_static_eplb_plan")
        return model

    module.initialize_model = initialize_model
    return module, initialize_model


def _remove_initializer_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in patch_model_loader_static_eplb.INITIALIZER_ALIASES:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_initialize_binds_after_construction_before_checkpoint_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = config_and_map(tmp_path)
    model = GenericMoE()
    module, _ = _initializer_module(model)
    _remove_initializer_aliases(monkeypatch)

    assert patch_model_loader_static_eplb.apply_to_module(module)
    loaded = module.initialize_model(config)

    assert loaded is model
    assert loaded._vllm_hcu_static_eplb_plan.layer_map(1) == (2, 0, 1, 2)
    assert (
        loaded.moe_layers[1].routed_experts._vllm_hcu_static_eplb_row
        == (2, 0, 1, 2)
    )
    assert patch_model_loader_static_eplb.apply_to_module(module) is False


def test_nested_initialize_binds_only_outer_model_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = config_and_map(tmp_path)
    model = GenericMoE()
    module = ModuleType(patch_model_loader_static_eplb.TARGET_MODULE)
    calls = []

    def initialize_model(
        vllm_config,
        *,
        prefix="",
        model_class=None,
        model_config=None,
    ):
        del model_class, model_config
        calls.append(prefix)
        if not prefix:
            return module.initialize_model(vllm_config, prefix="nested")
        return model

    module.initialize_model = initialize_model
    _remove_initializer_aliases(monkeypatch)

    static_eplb = importlib.import_module(
        "vllm_hcu.model_executor.layers.fused_moe.static_eplb"
    )
    original_bind = static_eplb.bind_static_eplb_plan
    bindings = []

    def bind_once(vllm_config, candidate):
        bindings.append(candidate)
        return original_bind(vllm_config, candidate)

    monkeypatch.setattr(static_eplb, "bind_static_eplb_plan", bind_once)
    patch_model_loader_static_eplb.apply_to_module(module)

    assert module.initialize_model(config) is model
    assert calls == ["", "nested"]
    assert bindings == [model]


@pytest.mark.parametrize(
    "load_format",
    ["auto", "pt", "safetensors", "npcache", "mistral"],
)
def test_static_loader_allows_audited_checkpoint_formats(
    tmp_path,
    load_format: str,
) -> None:
    config, _ = config_and_map(tmp_path)
    patch_model_loader_static_eplb.validate_static_loader(
        config,
        SimpleNamespace(load_format=load_format),
    )


@pytest.mark.parametrize(
    "load_format",
    ["dummy", "sharded_state", "tensorizer", "ipc_cache", "custom"],
)
def test_static_loader_rejects_formats_that_bypass_logical_loaders(
    tmp_path,
    load_format: str,
) -> None:
    config, _ = config_and_map(tmp_path)
    with pytest.raises(ValueError, match="loader format"):
        patch_model_loader_static_eplb.validate_static_loader(
            config,
            SimpleNamespace(load_format=load_format),
        )


def test_inactive_static_loader_does_not_restrict_format(tmp_path) -> None:
    config, _ = config_and_map(tmp_path)
    config.additional_config["hcu"]["expert_map_path"] = None
    patch_model_loader_static_eplb.validate_static_loader(
        config,
        SimpleNamespace(load_format="dummy"),
    )


def test_unsupported_format_fails_before_model_construction(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = config_and_map(tmp_path)
    config.load_config.load_format = "dummy"
    module = ModuleType(patch_model_loader_static_eplb.TARGET_MODULE)
    calls = []

    def initialize_model(
        vllm_config,
        *,
        prefix="",
        model_class=None,
        model_config=None,
    ):
        calls.append((vllm_config, prefix, model_class, model_config))
        return GenericMoE()

    module.initialize_model = initialize_model
    _remove_initializer_aliases(monkeypatch)
    patch_model_loader_static_eplb.apply_to_module(module)

    with pytest.raises(ValueError, match="loader format"):
        module.initialize_model(config)
    assert calls == []


def test_preimported_initializer_aliases_are_patched_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GenericMoE()
    module, original = _initializer_module(model)
    aliases = []
    for name in patch_model_loader_static_eplb.INITIALIZER_ALIASES:
        alias = ModuleType(name)
        alias.initialize_model = original
        monkeypatch.setitem(sys.modules, name, alias)
        aliases.append(alias)

    patch_model_loader_static_eplb.apply_to_module(module)

    assert all(alias.initialize_model is module.initialize_model for alias in aliases)


def test_stale_initializer_alias_rejects_patch_without_partial_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GenericMoE()
    module, original = _initializer_module(model)
    aliases = []
    for index, name in enumerate(
        patch_model_loader_static_eplb.INITIALIZER_ALIASES
    ):
        alias = ModuleType(name)
        alias.initialize_model = original if index else lambda: None
        monkeypatch.setitem(sys.modules, name, alias)
        aliases.append(alias)

    with pytest.raises(RuntimeError, match="alias mismatch"):
        patch_model_loader_static_eplb.apply_to_module(module)

    assert module.initialize_model is original
    assert aliases[1].initialize_model is original


def test_static_expert_mapping_removes_initial_redundant_checkpoint_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = importlib.import_module(patch_static_expert_mapping.TARGET_MODULE)
    cls = target.RoutedExperts
    original_descriptor = vars(cls)["make_expert_params_mapping"]
    monkeypatch.setattr(
        cls,
        "make_expert_params_mapping",
        original_descriptor,
    )
    monkeypatch.delattr(
        target,
        patch_static_expert_mapping._MARKER,
        raising=False,
    )
    assert patch_static_expert_mapping.apply_to_module(target)

    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
        bind_static_eplb_plan,
    )

    bind_static_eplb_plan(config, model)
    mapping = cls.make_expert_params_mapping(
        model,
        "gate_proj",
        "down_proj",
        "up_proj",
        3,
        1,
    )
    assert {entry[2] for entry in mapping} == {0, 1, 2}


def test_worker_inventory_arms_static_preload_and_mapping_boundaries() -> None:
    from vllm_hcu.patch.worker import _FRAMEWORK_CALLBACKS

    adapters = {spec.adapter for spec in _FRAMEWORK_CALLBACKS}
    assert (
        "vllm_hcu.patch.worker.framework_opt.patch_model_loader_static_eplb"
        in adapters
    )
    assert (
        "vllm_hcu.patch.worker.framework_opt.patch_static_expert_mapping"
        in adapters
    )
