import sys
from contextlib import contextmanager
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.core_fix import patch_dspark_smoke_layer


def _module(name: str, **symbols) -> ModuleType:
    module = ModuleType(name)
    for key, value in symbols.items():
        setattr(module, key, value)
    return module


def _dspark_module(get_model) -> ModuleType:
    module = _module(patch_dspark_smoke_layer.TARGET_MODULE, get_model=get_model)
    exec(
        "def load_dspark_model(target_model, vllm_config):\n"
        "    del target_model\n"
        "    return get_model(vllm_config=vllm_config, model_config='draft')\n",
        module.__dict__,
    )
    return module


def test_weight_debug_skip_context_is_nested_and_restored():
    from vllm_hcu.runtime_compat.weight_loading import (
        _WEIGHT_DEBUG_SKIP_DISABLED,
        disable_weight_debug_skip,
    )

    assert _WEIGHT_DEBUG_SKIP_DISABLED.get() is False
    with disable_weight_debug_skip():
        assert _WEIGHT_DEBUG_SKIP_DISABLED.get() is True
        with disable_weight_debug_skip():
            assert _WEIGHT_DEBUG_SKIP_DISABLED.get() is True
        assert _WEIGHT_DEBUG_SKIP_DISABLED.get() is True
    assert _WEIGHT_DEBUG_SKIP_DISABLED.get() is False


def test_dspark_draft_loading_disables_smoke_layer_limit(monkeypatch):
    active = False
    seen = []

    @contextmanager
    def disable_smoke_layer_limit():
        nonlocal active
        previous, active = active, True
        try:
            yield
        finally:
            active = previous

    def get_model(*, vllm_config, model_config=None, prefix="", load_config=None):
        seen.append((active, vllm_config, model_config, prefix, load_config))
        return "draft"

    module = _dspark_module(get_model)
    weight_utils = _module(
        patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE,
        disable_smoke_layer_limit=disable_smoke_layer_limit,
    )
    monkeypatch.setitem(
        sys.modules, patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE, weight_utils
    )

    assert patch_dspark_smoke_layer.apply_to_module(module) is True
    captured_loader = module.load_dspark_model
    assert captured_loader("target", "config") == "draft"
    assert seen == [(True, "config", "draft", "", None)]
    assert active is False
    assert patch_dspark_smoke_layer.apply_to_module(module) is False


def test_dspark_smoke_patch_disables_plugin_filter_when_base_has_no_smoke_api(
    monkeypatch,
):
    active = False
    seen = []

    @contextmanager
    def disable_weight_debug_skip():
        nonlocal active
        previous, active = active, True
        try:
            yield
        finally:
            active = previous

    def get_model(*, vllm_config, model_config=None, prefix="", load_config=None):
        seen.append(active)
        return vllm_config, model_config, prefix, load_config

    module = _dspark_module(get_model)
    weight_utils = _module(patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE)
    from vllm_hcu.runtime_compat import weight_loading

    monkeypatch.setattr(
        weight_loading, "disable_weight_debug_skip", disable_weight_debug_skip
    )
    monkeypatch.setitem(
        sys.modules, patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE, weight_utils
    )

    assert patch_dspark_smoke_layer.apply_to_module(module) is True
    assert module.load_dspark_model("target", "config") == (
        "config", "draft", "", None
    )
    assert seen == [True]
    assert active is False


def test_dspark_smoke_patch_rejects_non_callable_context_api(monkeypatch):
    def get_model(*, vllm_config, model_config=None, prefix="", load_config=None):
        return vllm_config, model_config, prefix, load_config

    module = _dspark_module(get_model)
    weight_utils = _module(
        patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE,
        disable_smoke_layer_limit=True,
    )
    monkeypatch.setitem(
        sys.modules, patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE, weight_utils
    )

    with pytest.raises(patch_dspark_smoke_layer.PatchCompatibilityError):
        patch_dspark_smoke_layer.apply_to_module(module)
    assert module.get_model is get_model


def test_dspark_smoke_patch_rejects_incompatible_loader_signature(monkeypatch):
    def get_model(vllm_config):
        return vllm_config

    module = _dspark_module(get_model)
    weight_utils = _module(
        patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE,
        disable_smoke_layer_limit=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, patch_dspark_smoke_layer.WEIGHT_UTILS_MODULE, weight_utils
    )

    with pytest.raises(patch_dspark_smoke_layer.PatchCompatibilityError):
        patch_dspark_smoke_layer.apply_to_module(module)
