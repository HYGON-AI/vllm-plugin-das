# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Explicit experimental DFly integration for Core v0.33.

No installed files are changed. Ordinary methods always use upstream entry
points. The standard platform dispatcher does not enable this module yet.
"""
from functools import wraps
import inspect
import sys
from types import FunctionType

_INSTALLED = False
_OWNED = ()


def install_v033_dfly():
    global _INSTALLED, _OWNED
    if _INSTALLED:
        for owner, name, value in _OWNED:
            if getattr(owner, name) is not value:
                raise RuntimeError(f"DFly compatibility owner changed: {name}")
        return False
    import vllm
    if not vllm.__version__.split("+")[0].startswith("0.33."):
        raise RuntimeError("This DFly compatibility path requires Core v0.33")
    from vllm import ModelRegistry
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    import vllm.v1.worker.gpu.spec_decode as factory
    import vllm.v1.worker.gpu.attn_utils as attn_utils
    from .dfly_config import HCUDFlySpeculativeConfig

    original = EngineArgs.create_speculative_config
    if tuple(inspect.signature(original).parameters) != (
            "self", "target_model_config", "target_parallel_config"):
        raise RuntimeError("DFly EngineArgs configuration ABI drift")
    upstream_config = original.__globals__.get("SpeculativeConfig")
    if not isinstance(upstream_config, type):
        raise RuntimeError("DFly requires the reviewed upstream config constructor")

    def construct(**kwargs):
        if kwargs.get("method") == "dfly":
            return HCUDFlySpeculativeConfig(**kwargs)
        return upstream_config(**kwargs)

    # Give only this function a private constructor dispatcher. Do not replace
    # the upstream class, mutate its Pydantic schema, or swap shared globals.
    private_globals = dict(original.__globals__, SpeculativeConfig=construct)
    create_config = FunctionType(original.__code__, private_globals,
                                 original.__name__, original.__defaults__,
                                 original.__closure__)
    create_config.__kwdefaults__ = original.__kwdefaults__
    wraps(original)(create_config)
    upstream_factory = factory.init_speculator

    @wraps(upstream_factory)
    def init_speculator(vllm_config, device, req_states):
        if isinstance(vllm_config.speculative_config, HCUDFlySpeculativeConfig):
            from .nvidia.dfly_speculator import HYV4DFlySpeculator
            return HYV4DFlySpeculator(vllm_config, device)
        return upstream_factory(vllm_config, device, req_states)

    upstream_shared = attn_utils.get_shared_kv_cache_layers
    upstream_add_shared = attn_utils.add_kv_sharing_layers_to_kv_cache_groups
    if tuple(inspect.signature(upstream_shared).parameters) != ("vllm_config",):
        raise RuntimeError("DFly KV-sharing discovery ABI drift")
    if tuple(inspect.signature(upstream_add_shared).parameters) != (
            "shared_kv_cache_layers", "kv_cache_groups", "runner_only_attn_layers"):
        raise RuntimeError("DFly KV-sharing group ABI drift")

    class DFlyKVSharingLayers(dict):
        """Tag discovery from the owned config, not arbitrary shared models."""

    @wraps(upstream_shared)
    def get_shared_kv_cache_layers(config):
        if not isinstance(config.speculative_config, HCUDFlySpeculativeConfig):
            return upstream_shared(config)
        # PR #68296 discovers AttentionLayerBase, not only dense Attention.
        # v0.33's dense-only check silently excludes MLAAttention aliases.
        layers = attn_utils.get_layers_from_vllm_config(
            config, attn_utils.AttentionLayerBase)
        return DFlyKVSharingLayers({
            name: target for name, layer in layers.items()
            if (target := getattr(layer, "kv_sharing_target_layer_name", None))
        })

    @wraps(upstream_add_shared)
    def add_kv_sharing_layers(shared_kv_cache_layers, kv_cache_groups,
                             runner_only_attn_layers=None):
        if not isinstance(shared_kv_cache_layers, DFlyKVSharingLayers):
            return upstream_add_shared(shared_kv_cache_layers, kv_cache_groups,
                                       runner_only_attn_layers)
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        groups = {name: group for group in kv_cache_groups
                  for name in group.layer_names}
        pending = []
        for name, target in shared_kv_cache_layers.items():
            if name == target or target not in groups:
                raise ValueError("DFly KV alias requires an allocated target layer")
            group = groups[target]
            if name in groups and groups[name] is not group:
                raise ValueError("DFly KV alias must share its target KV group")
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                target_spec = spec.kv_cache_specs[target]
                if name in spec.kv_cache_specs and spec.kv_cache_specs[name] != target_spec:
                    raise ValueError("DFly KV alias has an incompatible group spec")
            pending.append((name, target, group))
        # Both target and draft init_attn_backend call this helper. Unlike the
        # old list append, insertion must be idempotent. Heterogeneous groups
        # also need the alias's per-layer spec (new Core's dict does both).
        for name, target, group in pending:
            if name not in group.layer_names:
                group.layer_names.append(name)
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                specs = group.kv_cache_spec.kv_cache_specs
                specs[name] = specs[target]
            if runner_only_attn_layers is not None:
                runner_only_attn_layers.add(name)

    upstream_lookahead = VllmConfig.num_lookahead_tokens
    if not isinstance(upstream_lookahead, property):
        raise RuntimeError("DFly lookahead property ABI drift")

    def num_lookahead_tokens(config):
        if isinstance(config.speculative_config, HCUDFlySpeculativeConfig):
            return config.speculative_config.num_speculative_tokens + 1
        return upstream_lookahead.fget(config)

    lookahead = property(num_lookahead_tokens, doc=upstream_lookahead.__doc__)
    owned = [(EngineArgs, "create_speculative_config", create_config),
             (factory, "init_speculator", init_speculator),
             (attn_utils, "get_shared_kv_cache_layers", get_shared_kv_cache_layers),
             (attn_utils, "add_kv_sharing_layers_to_kv_cache_groups",
              add_kv_sharing_layers),
             (VllmConfig, "num_lookahead_tokens", lookahead)]
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None:
        if getattr(runner, "init_speculator", None) is not upstream_factory:
            raise RuntimeError("DFly runner speculator factory owner changed")
        owned.append((runner, "init_speculator", init_speculator))
    ModelRegistry.register_model(
        "HYV4DFlyModel", "vllm_hcu.models.hy_v4_flash.nvidia.dfly:HYV4DFlyForCausalLM")
    for owner, name, value in owned:
        setattr(owner, name, value)
    _OWNED = tuple(owned)
    _INSTALLED = True
    return True


def validate_dfly_runtime(vllm_config):
    from .dfly_config import HCUDFlySpeculativeConfig
    if not _INSTALLED or not isinstance(vllm_config.speculative_config,
                                         HCUDFlySpeculativeConfig):
        raise NotImplementedError("DFly requires the complete HCU v0.33 compatibility path")
    for owner, name, value in _OWNED:
        if getattr(owner, name) is not value:
            raise RuntimeError(f"DFly compatibility owner changed: {name}")
    if not vllm_config.model_config.enforce_eager:
        raise NotImplementedError("HCU DFly full-model graphs are not qualified")
