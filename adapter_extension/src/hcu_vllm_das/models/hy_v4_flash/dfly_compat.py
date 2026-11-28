# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, process-local Core033 DFly startup compatibility.

Calling install_compat is the opt-in; importing this module is inert. No Core
files, shared function globals, or retired implementation modules are changed.
Publication establishes ABI/ownership, NOT checkpoint/kernel/joint acceptance.
"""
import ast
from functools import wraps
import hashlib
from importlib import import_module
import inspect
import os
import sys
from threading import RLock
import textwrap

_ARCH = "HYV4DFlyModel"
_MODEL = "hcu_vllm_das.models.hy_v4_flash.dfly:HYV4DFlyForCausalLM"
_RUNNER = "vllm.v1.worker.gpu.model_runner"
_lock = RLock()
_state = None
_failure = None
_publishing = False


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if _state is not None or _publishing:
        _failure = "DFly publication inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


# SHA256 of reviewed standalone, dedented function source, as returned by
# inspect.getsource. Parsing an entire class instead retains different
# indentation inside multiline docstrings and produces incompatible hashes.
_CORE_ABI = {
    "register_model": ("vllm.model_executor.models.registry", "c09e84d88fcde8392fa7fc6aab89073983e51bc6813915afa2c5431ff85ee81c"),
    "create_speculative_config": ("vllm.engine.arg_utils", "a4507a83116fac2af6fd9bdf7b46beddfdf5e96e9bade6b046c96ba3dca0bbb2"),
    "init_speculator": ("vllm.v1.worker.gpu.spec_decode", "534513f9dedfb600306b4f8fde51907dd244b90a80abab23541cd6f3826058e4"),
    "get_shared_kv_cache_layers": ("vllm.v1.worker.gpu.attn_utils", "00520c5a9a0a058d85d1890a319e01a9b2ac3ec8bccb4b39084dca2d861cc5c8"),
    "add_kv_sharing_layers_to_kv_cache_groups": ("vllm.v1.worker.utils", "67a1990b1849bf4910cced5b4d8888c89611f2fb464bf7df2c43775f12a7ac6e"),
    "num_lookahead_tokens": ("vllm.config.vllm", "4b62657543437184f3ba518f75982f2c6931c66d61d7c8d9ecb7b91a70538726"),
}


def _source_abi(function):
    module, expected = _CORE_ABI[function.__name__]
    if function.__module__ != module:
        raise RuntimeError("DFly Core entry point has a foreign owner")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    digest = hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"DFly reviewed Core source ABI changed: {function.__name__}")


def _signature(function, names):
    if tuple(inspect.signature(function).parameters) != names:
        raise RuntimeError(f"DFly Core033 ABI drift: {function}")
    if hasattr(function, "__wrapped__"):
        raise RuntimeError("DFly refuses a previously wrapped Core entry point")
    _source_abi(function)


def _is_dfly(config):
    spec = getattr(config, "speculative_config", None)
    return getattr(spec, "method", None) == "dfly"


def _load_bindings():
    import vllm
    from vllm import ModelRegistry
    from vllm.model_executor.models import ModelRegistry as canonical_registry
    if ModelRegistry is not canonical_registry:
        raise RuntimeError("DFly canonical registry identity changed")
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from .dfly_config import HCUDFlySpeculativeConfig
    if vllm.__version__.split("+")[0] != "0.33.0":
        raise RuntimeError("DFly requires pinned Core 0.33.0")
    return (ModelRegistry, VllmConfig, EngineArgs,
            import_module("vllm.v1.worker.gpu.spec_decode"),
            import_module("vllm.v1.worker.gpu.attn_utils"),
            import_module("vllm.v1.worker.utils"), HCUDFlySpeculativeConfig)


class _DFlySharing(dict):
    """Only DFly-discovered MLA aliases take the owned group-add path."""


def _add_dfly_sharing(aliases, groups, runner_only=None):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    by_name = {}
    for group in groups:
        for name in group.layer_names:
            if name in by_name and by_name[name] is not group:
                raise ValueError("DFly layer occurs in multiple KV groups")
            by_name[name] = group
    pending = []
    # Entire transaction validated before any group/set mutation.
    for name, target in aliases.items():
        if name == target or target not in by_name:
            raise ValueError("DFly KV alias requires an allocated distinct target")
        group = by_name[target]
        if name in by_name and by_name[name] is not group:
            raise ValueError("DFly alias already belongs to a different KV group")
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            if target not in spec.kv_cache_specs:
                raise ValueError("DFly target has no per-layer KV specification")
            if name in spec.kv_cache_specs and spec.kv_cache_specs[name] != spec.kv_cache_specs[target]:
                raise ValueError("DFly alias KV specification mismatch")
        pending.append((name, target, group))
    for name, target, group in pending:
        if name not in group.layer_names:
            group.layer_names.append(name)
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            specs = group.kv_cache_spec.kv_cache_specs
            specs[name] = specs[target]
        if runner_only is not None:
            runner_only.add(name)


def _owned_health():
    if _failure is not None:
        raise RuntimeError(_failure)
    if _state is None:
        raise RuntimeError("DFly compatibility has not been installed")
    pid, bindings, registry, entry, register, factory_wrapper, sources = _state
    live = _load_bindings()
    if len(live) != len(sources) or any(a is not b for a, b in zip(live, sources)):
        raise RuntimeError("DFly Core/config module bindings changed")
    if pid != os.getpid():
        raise RuntimeError("DFly compatibility cannot be inherited across fork")
    for owner, name, value in bindings:
        if getattr(owner, name) is not value:
            raise RuntimeError(f"DFly compatibility owner changed: {name}")
    for owner, _, _ in bindings:
        module_name = getattr(owner, "__name__", "")
        if module_name.startswith("vllm.") and not isinstance(owner, type) and sys.modules.get(module_name) is not owner:
            raise RuntimeError("DFly Core module identity changed")
    if registry.models.get(_ARCH) is not entry:
        raise RuntimeError("DFly model registration owner changed")
    if (entry.module_name, entry.class_name) != tuple(_MODEL.split(":")):
        raise RuntimeError("DFly registry target changed")
    method = registry.register_model
    if getattr(method, "__func__", method) is not register:
        raise RuntimeError("DFly registry publisher changed")
    runner = sys.modules.get(_RUNNER)
    if runner is not None and getattr(runner, "init_speculator", None) is not factory_wrapper:
        raise RuntimeError("DFly runner factory alias is not owned")
    return {"installed": True, "pid": pid, "architecture": _ARCH,
            "wrapped_points": tuple(name for _, name, _ in bindings)}


def check_health() -> dict:
    """Return publication metadata; raise on uninstalled/fork/drift/partial state.

    This is an ownership/ABI check, not an execution or model support claim.
    A detected failure is latched: restart rather than silently taking over.
    """
    global _failure
    with _lock:
        try:
            return _owned_health()
        except BaseException as error:
            if _state is not None and _failure is None:
                _failure = str(error)
            raise


def validate_dfly_runtime(vllm_config) -> None:
    """Non-DFly is inert. DFly requires healthy owned config and eager V2.

    Target/HiSparse/joint admission remains independently enforced; this never
    edits a guard or certifies native attention/kernel/checkpoint acceptance.
    """
    if not _is_dfly(vllm_config):
        return None
    check_health()
    from .dfly_config import HCUDFlySpeculativeConfig
    if not isinstance(vllm_config.speculative_config, HCUDFlySpeculativeConfig):
        raise ValueError("DFly requires the owned speculative config")
    from .fullgraph_runner import _is_hisparse_connector

    # EngineArgs temporarily holds the connector before its resolved HiSparse config.
    hisparse = (os.environ.get("HCU_VLLM_HY4_HISPARSE") == "1"
                and _is_hisparse_connector(vllm_config))
    if getattr(vllm_config.attention_config, "hisparse_config", None) is not None and not hisparse:
        raise NotImplementedError("HiSparse + DFly joint runtime remains unqualified")
    p = vllm_config.parallel_config
    target = vllm_config.speculative_config.target_model_config
    if target.hf_text_config.model_type == "hy_v4":
        from .dfly_topology import validate_standard_dfly_runtime_parallel
        # Draft configs deliberately disable EP; validate the immutable target
        # owner rather than treating the dense draft as an expert participant.
        validate_standard_dfly_runtime_parallel(
            p, vllm_config.speculative_config.target_parallel_config,
            enforce_eager=target.enforce_eager)
    elif any(getattr(p, field, 1) != 1 for field in (
        "data_parallel_size", "pipeline_parallel_size",
        "prefill_context_parallel_size", "decode_context_parallel_size")):
        raise ValueError("DFly requires DP/PP/PCP/DCP=1")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("DFly requires V2 model runner")
    if (getattr(vllm_config, "lora_config", None) is not None
            or (getattr(vllm_config, "kv_transfer_config", None) is not None and not hisparse)):
        raise ValueError("DFly does not admit LoRA or KV transfer")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
        raise ValueError("DFly requires BF16 or fp8_ds_mla KV cache")
    return None


def install_compat() -> bool:
    """Explicit opt-in. True first commit; False healthy repeat; errors latch.

    All known entry points are preflighted before publication. Non-DFly calls
    delegate the exact original once, with no preliminary CLI mutation. A failed
    publication rolls back owned writes but remains unusable in this process.
    """
    global _state, _failure, _publishing
    with _lock:
        if _failure is not None:
            raise RuntimeError(_failure)
        if _state is not None:
            check_health()
            return False
        if _publishing:
            _failure = "recursive DFly compatibility publication"
            raise RuntimeError(_failure)
        _publishing = True
        changed = []
        registry = None
        entry = None
        try:
            sources = _load_bindings()
            registry, config_cls, args_cls, factory, attn, utils, spec_cls = sources
            create = args_cls.create_speculative_config
            init = factory.init_speculator
            discover = attn.get_shared_kv_cache_layers
            add = utils.add_kv_sharing_layers_to_kv_cache_groups
            lookahead = config_cls.num_lookahead_tokens
            _signature(create, ("self", "target_model_config", "target_parallel_config"))
            _signature(init, ("vllm_config", "device", "req_states"))
            _signature(discover, ("vllm_config",))
            _signature(add, ("shared_kv_cache_layers", "kv_cache_groups", "runner_only_attn_layers"))
            if not isinstance(lookahead, property) or lookahead.fget is None:
                raise RuntimeError("DFly lookahead property ABI drift")
            _signature(lookahead.fget, ("self",))
            if attn.add_kv_sharing_layers_to_kv_cache_groups is not add:
                raise RuntimeError("DFly KV helper alias already has another owner")
            if _ARCH in registry.models:
                raise RuntimeError("DFly model already has a registry owner")
            runner = sys.modules.get(_RUNNER)
            if runner is not None and getattr(runner, "init_speculator", None) is not init:
                raise RuntimeError("DFly runner alias already has another owner")
            register = registry.register_model
            _signature(register, ("model_arch", "model_cls"))

            @wraps(create)
            def create_config(self, target_model_config, target_parallel_config):
                supplied = self.speculative_config
                requested = self.spec_method
                if requested != "dfly" and not (
                    supplied is not None and supplied.get("method") == "dfly"
                ):
                    return create(self, target_model_config, target_parallel_config)
                # Owned Core033 CLI control flow. Do not call original again.
                for flag, key, value in (
                    ("--spec-method", "method", self.spec_method),
                    ("--spec-model", "model", self.spec_model),
                    ("--spec-tokens", "num_speculative_tokens", self.spec_tokens),
                ):
                    if value is None:
                        continue
                    if self.speculative_config is None:
                        self.speculative_config = {}
                    if key in self.speculative_config:
                        raise ValueError(f"{flag} and --speculative-config['{key}'] are mutually exclusive")
                    self.speculative_config[key] = value
                self.speculative_config = {
                    k.replace("-", "_"): v for k, v in self.speculative_config.items()
                }
                self.speculative_config.update({
                    "target_model_config": target_model_config,
                    "target_parallel_config": target_parallel_config,
                })
                return spec_cls(**self.speculative_config)

            @wraps(init)
            def init_speculator(vllm_config, device, req_states):
                if not _is_dfly(vllm_config):
                    return init(vllm_config, device, req_states)
                validate_dfly_runtime(vllm_config)
                from .dfly_speculator import HYV4DFlySpeculator
                return HYV4DFlySpeculator(vllm_config, device)

            @wraps(discover)
            def get_shared_kv_cache_layers(vllm_config):
                if not _is_dfly(vllm_config):
                    return discover(vllm_config)
                validate_dfly_runtime(vllm_config)
                layers = attn.get_layers_from_vllm_config(vllm_config, attn.AttentionLayerBase)
                return _DFlySharing({name: target for name, layer in layers.items()
                    if (target := getattr(layer, "kv_sharing_target_layer_name", None))})

            @wraps(add)
            def add_shared(shared_kv_cache_layers, kv_cache_groups, runner_only_attn_layers=None):
                if not isinstance(shared_kv_cache_layers, _DFlySharing):
                    return add(shared_kv_cache_layers, kv_cache_groups, runner_only_attn_layers)
                check_health()
                return _add_dfly_sharing(shared_kv_cache_layers, kv_cache_groups, runner_only_attn_layers)

            def num_lookahead_tokens(self):
                if not _is_dfly(self):
                    return lookahead.fget(self)
                validate_dfly_runtime(self)
                return self.speculative_config.num_speculative_tokens + 1

            pending = [
                (args_cls, "create_speculative_config", create_config),
                (factory, "init_speculator", init_speculator),
                (attn, "get_shared_kv_cache_layers", get_shared_kv_cache_layers),
                (utils, "add_kv_sharing_layers_to_kv_cache_groups", add_shared),
                (attn, "add_kv_sharing_layers_to_kv_cache_groups", add_shared),
                (config_cls, "num_lookahead_tokens", property(num_lookahead_tokens,
                    lookahead.fset, lookahead.fdel, lookahead.__doc__)),
            ]
            if runner is not None:
                pending.append((runner, "init_speculator", init_speculator))
            for owner, name, value in pending:
                previous = getattr(owner, name)
                setattr(owner, name, value)
                changed.append((owner, name, previous, value))
            registry.register_model(_ARCH, _MODEL)
            entry = registry.models[_ARCH]
            _state = (os.getpid(), tuple(pending), registry, entry,
                      getattr(register, "__func__", register), init_speculator, sources)
            check_health()
            return True
        except BaseException as error:
            # Preflight failure is retryable only if nothing was published.
            if changed:
                _failure = f"DFly partial publication failed: {error}; restart process"
                for owner, name, previous, value in reversed(changed):
                    if getattr(owner, name) is value:
                        setattr(owner, name, previous)
                if registry is not None and entry is not None and registry.models.get(_ARCH) is entry:
                    registry.models.pop(_ARCH)
                _state = None
            raise
        finally:
            _publishing = False
