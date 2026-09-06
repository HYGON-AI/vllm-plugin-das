# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_model_loader_static_eplb


def _initializer_module() -> ModuleType:
    module = ModuleType(patch_model_loader_static_eplb.TARGET_MODULE)

    def initialize_model(
        vllm_config: object,
        *,
        prefix: str = "",
        model_class: type | None = None,
        model_config: object | None = None,
    ) -> object:
        del vllm_config, prefix, model_class, model_config
        return object()

    module.initialize_model = initialize_model
    return module


def test_initializer_binds_the_constructed_model_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _initializer_module()
    events: list[tuple[str, object]] = []

    def bind(vllm_config: object, model: object) -> None:
        events.append(("bind", vllm_config))
        assert model is result_holder["model"]

    result_holder: dict[str, object] = {}
    original = module.initialize_model

    def initialize_model(
        vllm_config: object,
        *,
        prefix: str = "",
        model_class: type | None = None,
        model_config: object | None = None,
    ) -> object:
        model = original(
            vllm_config,
            prefix=prefix,
            model_class=model_class,
            model_config=model_config,
        )
        result_holder["model"] = model
        events.append(("initialize", vllm_config))
        return model

    module.initialize_model = initialize_model
    monkeypatch.setattr(patch_model_loader_static_eplb, "bind_static_eplb_plan", bind)

    assert patch_model_loader_static_eplb.apply_to_module(module) is True
    config = object()
    model = module.initialize_model(config)

    assert model is result_holder["model"]
    assert events == [("initialize", config), ("bind", config)]
    assert patch_model_loader_static_eplb.apply_to_module(module) is False


def test_initializer_patch_updates_an_existing_verified_loader_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _initializer_module()
    alias = ModuleType("vllm.model_executor.model_loader.base_loader")
    alias.initialize_model = module.initialize_model
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    assert patch_model_loader_static_eplb.apply_to_module(module) is True
    assert alias.initialize_model is module.initialize_model


def test_initializer_patch_rejects_an_existing_stale_loader_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _initializer_module()
    alias = ModuleType("vllm.model_executor.model_loader.base_loader")
    alias.initialize_model = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="already-imported alias",
    ):
        patch_model_loader_static_eplb.apply_to_module(module)
