"""Ensure GEMM preloading preserves custom loader lifecycle and contexts."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.runtime_compat.kimi_k3_loading import (
    is_kimi_k3_config,
    preload_before_weight_loading,
)


@pytest.mark.parametrize("fail", [False, True])
def test_preload_keeps_loader_override_dtype_and_restores_on_error(fail):
    events = []
    model = object()

    class Loader:
        def load_model(self):
            events.append("custom_setup")
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                self.load_weights(model, None)
                events.append("finalize")
                return model
            finally:
                torch.set_default_dtype(old_dtype)

        def load_weights(self, actual_model, config):
            assert actual_model is model
            assert torch.get_default_dtype() == torch.float64
            events.append("weights")
            if fail:
                raise RuntimeError("load failed")

    loader = Loader()
    original = loader.load_weights

    def preload(actual_model):
        assert actual_model is model
        assert torch.get_default_dtype() == torch.float64
        events.append("preload")

    if fail:
        with pytest.raises(RuntimeError, match="load failed"):
            with preload_before_weight_loading(loader, preload):
                loader.load_model()
    else:
        with preload_before_weight_loading(loader, preload):
            assert loader.load_model() is model
    assert loader.load_weights == original
    assert "load_weights" not in vars(loader)
    assert events == ["custom_setup", "preload", "weights"] + (
        [] if fail else ["finalize"]
    )


def test_preload_restores_instance_override():
    callback = lambda model, config: None
    loader = SimpleNamespace(load_weights=callback)
    with preload_before_weight_loading(loader, lambda model: None):
        assert loader.load_weights is not callback
    assert loader.load_weights is callback


@pytest.mark.parametrize("arch, expected", [
    ("KimiK3ForConditionalGeneration", True),
    ("Qwen3ForCausalLM", False),
])
def test_preload_scope_is_kimi_only(arch, expected):
    config = SimpleNamespace(model_config=SimpleNamespace(architectures=[arch]))
    assert is_kimi_k3_config(config) is expected
