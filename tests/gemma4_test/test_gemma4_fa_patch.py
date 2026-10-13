# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model registration and model-local backend injection contracts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.models import gemma4


@pytest.fixture
def model_module(monkeypatch):
    import vllm.config

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(backend=None)),
    )

    class Attention:
        def __init__(
            self,
            num_heads,
            head_size,
            scale,
            *,
            per_layer_sliding_window=None,
            attn_backend=None,
        ):
            self.backend = attn_backend
            self.head_size = head_size

    module = ModuleType(gemma4._GEMMA4_MODULE)
    module.Attention = Attention
    return module, Attention


@pytest.mark.parametrize(
    "head_size,window,enabled",
    [
        (512, None, True),
        (256, None, False),
        (256, 1024, False),
        (512, 1024, False),
    ],
)
def test_backend_injection_is_model_local_and_idempotent(
    model_module, head_size, window, enabled
):
    module, original = model_module
    assert gemma4._use_gemma4_fa(module)
    assert not gemma4._use_gemma4_fa(module)
    layer = module.Attention(16, head_size, 1.0, per_layer_sliding_window=window)
    expected = gemma4.Gemma4FAAttentionBackend if enabled else None
    assert layer.backend is expected
    assert original(16, 512, 1.0).backend is None
    explicit = object()
    assert module.Attention(16, 512, 1.0, attn_backend=explicit).backend is explicit


def test_conflicting_config_backend_fails_for_target_layers_only(
    monkeypatch, model_module
):
    import vllm.config

    module, _ = model_module
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(backend="FLASH_ATTN")),
    )
    gemma4._use_gemma4_fa(module)
    with pytest.raises(ValueError, match="conflicts"):
        module.Attention(16, 512, 1.0)
    assert module.Attention(8, 256, 1.0, per_layer_sliding_window=1024).backend is None


def test_post_import_registration_is_deferred_and_idempotent(monkeypatch, model_module):
    import vllm_hcu.patch as patches
    from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
    from vllm_hcu.patch.runtime_state import PatchRegistry

    module, _ = model_module
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    monkeypatch.setattr(patches, "register_post_import", coordinator.register_callback)
    monkeypatch.delitem(sys.modules, gemma4._GEMMA4_MODULE, raising=False)
    gemma4.register_attention()
    assert gemma4._GEMMA4_MODULE not in sys.modules
    # Simulate completed target import, then re-enter registration.
    monkeypatch.setitem(sys.modules, gemma4._GEMMA4_MODULE, module)
    gemma4.register_attention()
    injected = module.Attention
    gemma4.register_attention()
    assert module.Attention is injected
    assert module.Attention(16, 512, 1.0).backend is gemma4.Gemma4FAAttentionBackend


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
    ],
)
def test_existing_model_registration_gates_the_hook(monkeypatch, raw, enabled):
    registration, calls = _model_registration(monkeypatch)
    if raw is None:
        monkeypatch.delenv("VLLM_HCU_GEMMA4_USE_FA", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_GEMMA4_USE_FA", raw)
    registration.register_model()
    assert len(calls) == int(enabled)


def test_invalid_switch_fails_before_model_registration(monkeypatch):
    registration, calls = _model_registration(monkeypatch)
    monkeypatch.setenv("VLLM_HCU_GEMMA4_USE_FA", "maybe")
    with pytest.raises(ValueError, match="VLLM_HCU_GEMMA4_USE_FA"):
        registration.register_model()
    assert not calls


def _model_registration(monkeypatch):
    import vllm

    calls = []
    monkeypatch.setattr(
        vllm,
        "ModelRegistry",
        SimpleNamespace(register_model=lambda *args: None),
        raising=False,
    )
    hy_config = ModuleType("vllm_hcu.models.hy_v4.config")
    hy_config.register_hy_v4_config = lambda: None
    monkeypatch.setitem(sys.modules, hy_config.__name__, hy_config)
    monkeypatch.setattr(gemma4, "register_attention", lambda: calls.append("gemma4"))
    path = Path(gemma4.__file__).with_name("__init__.py")
    spec = importlib.util.spec_from_file_location("_gemma4_registration_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls
