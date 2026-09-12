# SPDX-License-Identifier: Apache-2.0
"""Reject checkpoint formats that bypass logical expert loaders."""
import functools
import sys

from ._common import load_exact_module, require_exact_signature, PatchCompatibilityError
from .patch_model_loader_static_eplb import validate_static_loader

TARGET_MODULE = "vllm.model_executor.model_loader"
PATCH_ID = "worker.framework_opt.model_loader.static_eplb_gate"
TARGETS = (f"{TARGET_MODULE}.get_model_loader",)
_MARKER = "_vllm_hcu_static_loader_gate"
_ALIASES = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def apply_to_module(module):
    target = load_exact_module(TARGET_MODULE, module)
    original = target.get_model_loader
    if getattr(target, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("Static EPLB loader gate marker is stale")
        return False
    require_exact_signature(original, TARGETS[0], positional=("load_config",))
    aliases = []
    for name in _ALIASES:
        alias = sys.modules.get(name)
        if alias is None:
            continue
        value = getattr(alias, "get_model_loader", None)
        if value is None and getattr(getattr(alias, "__spec__", None), "_initializing", False):
            continue
        if value is not original:
            raise PatchCompatibilityError(f"Static EPLB loader factory alias mismatch: {name}")
        aliases.append(alias)

    @functools.wraps(original)
    def get_model_loader(load_config):
        loader = original(load_config)
        load_model = loader.load_model
        @functools.wraps(load_model)
        def load(vllm_config, model_config, prefix=""):
            validate_static_loader(vllm_config, load_config)
            if getattr(vllm_config.parallel_config, "_vllm_hcu_expert_map_path", None):
                if type(loader) is not target.DefaultModelLoader:
                    raise ValueError("Static EPLB requires the current DefaultModelLoader")
            return load_model(vllm_config, model_config, prefix)
        loader.load_model = load
        return loader
    setattr(get_model_loader, _MARKER, True)
    target.get_model_loader = get_model_loader
    for alias in aliases:
        alias.get_model_loader = get_model_loader
    setattr(target, _MARKER, True)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
