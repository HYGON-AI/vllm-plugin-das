"""Preload Kimi GEMMs without replacing a loader's model lifecycle."""

from contextlib import contextmanager
from functools import wraps


def is_kimi_k3_config(vllm_config):
    model_config = getattr(vllm_config, "model_config", None)
    return any(
        isinstance(arch, str) and arch.startswith("KimiK3")
        for arch in (getattr(model_config, "architectures", ()) or ())
    )


@contextmanager
def preload_before_weight_loading(loader, preload):
    """Keep load_model overrides, device/dtype contexts and finalization intact."""
    original = loader.load_weights
    had_override = "load_weights" in vars(loader)
    previous = vars(loader).get("load_weights")

    @wraps(original)
    def load_weights(model, model_config):
        preload(model)
        return original(model, model_config)

    loader.load_weights = load_weights
    try:
        yield
    finally:
        if had_override:
            loader.load_weights = previous
        else:
            del loader.load_weights
