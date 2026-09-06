# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_model_loader_static_eplb


@pytest.fixture(autouse=True)
def _isolate_initializer_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep alias-validation tests independent of prior vLLM imports."""

    for module_name in (
        *patch_model_loader_static_eplb._INITIALIZER_ALIASES,
        *patch_model_loader_static_eplb._LOADER_FACTORY_ALIASES,
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)


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
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(_vllm_hcu_expert_map_path="map.json")
    )
    model = module.initialize_model(config)

    assert model is result_holder["model"]
    assert events == [("initialize", config), ("bind", config)]
    assert patch_model_loader_static_eplb.apply_to_module(module) is False


@pytest.mark.parametrize(
    "alias_name",
    patch_model_loader_static_eplb._INITIALIZER_ALIASES,
)
def test_initializer_patch_updates_every_existing_verified_loader_alias(
    monkeypatch: pytest.MonkeyPatch,
    alias_name: str,
) -> None:
    module = _initializer_module()
    alias = ModuleType(alias_name)
    alias.initialize_model = module.initialize_model
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    assert patch_model_loader_static_eplb.apply_to_module(module) is True
    assert alias.initialize_model is module.initialize_model


@pytest.mark.parametrize(
    "alias_name",
    patch_model_loader_static_eplb._INITIALIZER_ALIASES,
)
def test_initializer_patch_rejects_every_existing_stale_loader_alias(
    monkeypatch: pytest.MonkeyPatch,
    alias_name: str,
) -> None:
    module = _initializer_module()
    alias = ModuleType(alias_name)
    alias.initialize_model = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, alias.__name__, alias)

    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="already-imported alias",
    ):
        patch_model_loader_static_eplb.apply_to_module(module)


def test_initializer_no_map_delegates_without_running_static_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _initializer_module()
    original = module.initialize_model
    sentinel = object()

    def initialize_model(
        vllm_config: object,
        *,
        prefix: str = "",
        model_class: type | None = None,
        model_config: object | None = None,
    ) -> object:
        assert prefix == "prefix"
        assert model_class is int
        assert model_config is sentinel
        return original(
            vllm_config,
            prefix=prefix,
            model_class=model_class,
            model_config=model_config,
        )

    module.initialize_model = initialize_model
    monkeypatch.setattr(
        patch_model_loader_static_eplb,
        "bind_static_eplb_plan",
        lambda *args, **kwargs: pytest.fail("no-map initialization must not bind"),
    )
    assert patch_model_loader_static_eplb.apply_to_module(module) is True

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(_vllm_hcu_expert_map_path=None)
    )
    result = module.initialize_model(
        config,
        prefix="prefix",
        model_class=int,
        model_config=sentinel,
    )

    assert type(result) is object


def test_nested_static_initializer_binds_only_the_public_outer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _initializer_module()
    alias_name = "vllm.model_executor.models.mllama4"
    alias = ModuleType(alias_name)
    bound_models: list[object] = []

    class Llama4ForCausalLM:
        pass

    class Llama4ForConditionalGeneration:
        def __init__(self, language_model: object) -> None:
            self.language_model = language_model

    def initialize_model(
        vllm_config: object,
        *,
        prefix: str = "",
        model_class: type | None = None,
        model_config: object | None = None,
    ) -> object:
        del prefix, model_config
        if model_class is Llama4ForCausalLM:
            return Llama4ForCausalLM()
        language_model = alias.initialize_model(
            vllm_config,
            model_class=Llama4ForCausalLM,
        )
        return Llama4ForConditionalGeneration(language_model)

    module.initialize_model = initialize_model
    alias.initialize_model = initialize_model
    monkeypatch.setitem(sys.modules, alias_name, alias)
    monkeypatch.setattr(
        patch_model_loader_static_eplb,
        "bind_static_eplb_plan",
        lambda _config, model: bound_models.append(model),
    )
    assert patch_model_loader_static_eplb.apply_to_module(module) is True

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(_vllm_hcu_expert_map_path="map.json")
    )
    model = module.initialize_model(config)

    assert isinstance(model, Llama4ForConditionalGeneration)
    assert isinstance(model.language_model, Llama4ForCausalLM)
    assert bound_models == [model]


class _FakeLoader:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load_model(
        self,
        vllm_config: object,
        model_config: object,
        prefix: str = "",
    ) -> object:
        del vllm_config, model_config, prefix
        self.events.append("checkpoint-consumed")
        return self


class _DefaultModelLoader(_FakeLoader):
    pass


class _BitsAndBytesModelLoader(_FakeLoader):
    pass


class _RunaiModelStreamerLoader(_FakeLoader):
    pass


class _ShardedStateLoader(_FakeLoader):
    pass


class _TensorizerLoader(_FakeLoader):
    tensorizer_config = object()


class _DummyModelLoader(_FakeLoader):
    pass


class _ModelExpressModelLoader(_FakeLoader):
    pass


def _model_loader_entrypoint_module(
    loader: _FakeLoader,
    original_calls: list[tuple[object, object, str, object]],
) -> ModuleType:
    module = ModuleType("vllm.model_executor.model_loader")

    def get_model_loader(load_config: object) -> _FakeLoader:
        del load_config
        return loader

    def get_model(
        *,
        vllm_config: object,
        model_config: object | None = None,
        prefix: str = "",
        load_config: object | None = None,
    ) -> object:
        original_calls.append((vllm_config, model_config, prefix, load_config))
        selected = module.get_model_loader(load_config or vllm_config.load_config)
        if model_config is None:
            model_config = vllm_config.model_config
        return selected.load_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )

    module.get_model_loader = get_model_loader
    module.get_model = get_model
    module.DefaultModelLoader = _DefaultModelLoader
    module.BitsAndBytesModelLoader = _BitsAndBytesModelLoader
    module.RunaiModelStreamerLoader = _RunaiModelStreamerLoader
    module.ShardedStateLoader = _ShardedStateLoader
    module.TensorizerLoader = _TensorizerLoader
    module.DummyModelLoader = _DummyModelLoader
    module.ModelExpressModelLoader = _ModelExpressModelLoader
    module._LOAD_FORMAT_TO_MODEL_LOADER = {"test": type(loader)}
    return module


def _loader_config(*, static: bool) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            _vllm_hcu_expert_map_path="map.json" if static else None
        ),
        load_config=SimpleNamespace(load_format="test"),
        model_config=object(),
    )


def _install_bound_initializer(monkeypatch: pytest.MonkeyPatch) -> None:
    utils = ModuleType(patch_model_loader_static_eplb.TARGET_MODULE)
    utils.initialize_model = lambda *args, **kwargs: None
    setattr(
        utils.initialize_model,
        patch_model_loader_static_eplb._WRAPPER_MARKER,
        True,
    )
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


def test_model_entrypoint_no_map_delegates_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_calls: list[tuple[object, object, str, object]] = []

    class ExternalLoader(_DefaultModelLoader):
        pass

    loader = ExternalLoader(events)
    module = _model_loader_entrypoint_module(loader, original_calls)
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True
    config = _loader_config(static=False)
    model_config = object()
    load_config = object()

    result = module.get_model(
        vllm_config=config,
        model_config=model_config,
        prefix="prefix",
        load_config=load_config,
    )

    assert result is loader
    assert original_calls == [(config, model_config, "prefix", load_config)]
    assert events == ["checkpoint-consumed"]


@pytest.mark.parametrize(
    "loader_type",
    (_ShardedStateLoader, _DummyModelLoader, _ModelExpressModelLoader),
)
def test_static_model_entrypoint_rejects_unsafe_builtin_loader_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
    loader_type: type[_FakeLoader],
) -> None:
    events: list[str] = []
    loader = loader_type(events)
    module = _model_loader_entrypoint_module(loader, [])
    _install_bound_initializer(monkeypatch)
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True

    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="does not support static EPLB direct loading",
    ):
        module.get_model(vllm_config=_loader_config(static=True))

    assert events == []


def test_static_model_entrypoint_rejects_external_loader_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ExternalLoader(_DefaultModelLoader):
        pass

    module = _model_loader_entrypoint_module(ExternalLoader(events), [])
    _install_bound_initializer(monkeypatch)
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True

    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="ExternalLoader.+does not support",
    ):
        module.get_model(vllm_config=_loader_config(static=True))

    assert events == []


def test_static_loader_factory_gate_covers_primary_gpu_runner_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    loader = _ShardedStateLoader(events)
    module = _model_loader_entrypoint_module(loader, [])
    _install_bound_initializer(monkeypatch)
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True
    config = _loader_config(static=True)

    selected_loader = module.get_model_loader(config.load_config)
    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="does not support static EPLB direct loading",
    ):
        selected_loader.load_model(
            vllm_config=config,
            model_config=config.model_config,
        )

    assert events == []


def test_vllm_tensorized_loader_fails_before_direct_deserialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    module = _model_loader_entrypoint_module(_TensorizerLoader(events), [])
    _install_bound_initializer(monkeypatch)
    monkeypatch.setattr(
        patch_model_loader_static_eplb,
        "_is_vllm_tensorized_loader",
        lambda loader: True,
    )
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True

    with pytest.raises(
        patch_model_loader_static_eplb.PatchCompatibilityError,
        match="vLLM-tensorized Tensorizer",
    ):
        module.get_model(vllm_config=_loader_config(static=True))

    assert events == []


@pytest.mark.parametrize(
    "loader_type",
    (
        _DefaultModelLoader,
        _BitsAndBytesModelLoader,
        _RunaiModelStreamerLoader,
        _TensorizerLoader,
    ),
)
def test_static_model_entrypoint_allows_only_audited_common_loaders(
    monkeypatch: pytest.MonkeyPatch,
    loader_type: type[_FakeLoader],
) -> None:
    events: list[str] = []
    loader = loader_type(events)
    module = _model_loader_entrypoint_module(loader, [])
    _install_bound_initializer(monkeypatch)
    monkeypatch.setattr(
        patch_model_loader_static_eplb,
        "_is_vllm_tensorized_loader",
        lambda candidate: False,
    )
    assert patch_model_loader_static_eplb.apply_entrypoint_to_module(module) is True
    config = _loader_config(static=True)

    result = module.get_model(
        vllm_config=config,
        model_config=config.model_config,
        prefix="prefix",
    )

    assert result is loader
    assert events == ["checkpoint-consumed"]
