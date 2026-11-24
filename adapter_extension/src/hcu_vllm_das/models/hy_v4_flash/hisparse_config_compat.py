# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit HCU Flash/Core033 HiSparse *configuration* construction boundary.

Import is inert. install_compat owns only EngineArgs.create_engine_config; it
never changes Core files, platforms, function globals or VllmConfig.__post_init__.
The real factory and real ModelConfig construction each run once. A conservative pinned HF-metadata preflight selects candidates. A private
EngineArgs copy captures the latter's result to admit only actual Flash models,
including the VL outer config. Non-Flash delegates the original input exactly once. Callable HF overrides and
remote-code configs remain native-only; the selector never executes them twice.

Only explicit eager V2, BF16, configuration/planner metadata,
DP/PP/PCP/DCP=1, no DBO/LoRA/offloading is admitted. The separately installed
planner owner may admit its explicit external connector and genuine DFly5
configuration to METADATA planning, never joint execution. This is deliberately NOT
HiSparse+DFly admission or kernel/checkpoint/engine qualification. Activation
belongs to the caller. Direct VllmConfig construction keeps Core's CUDA guard.

IMPORTANT: this pin's host-pool planner requires HiSparseConnector.host_pool_gib.
The requested no-connector boundary cannot start a HiSparse engine. The config
validator is separate from validate_hisparse_runtime, which fails closed with
that concrete downstream blocker. A successful factory/health check is NOT
runtime admission. The separately installed hisparse_planner_compat owns an
explicit external connector/builder for planner metadata. Its execution remains
blocked; post-init compatibility and worker/device paths must not be skipped.

The reviewed Core has three HiSparse-dependent sites: runner selection, the
inline constraints, and late HMA validation. Explicit V2 makes runner selection
identical while the private field is hidden. The original post-init performs ALL
other normalization/platform/MM/runner/HMA effects; we replay the constraints
against its final result. No post-init or platform hook is run a second time.
"""

import ast
import hashlib
import inspect
import os
import sys
import textwrap
import weakref
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from importlib import import_module
from threading import RLock

_ARCHES = frozenset(("HYV4FlashForCausalLM", "HYV4FlashVLForConditionalGeneration"))
# Core 4ba787ec9959e566d3b587a796dff07673324da5. Entire class ASTs pin
# transitive runner/HMA helpers as well as the public entry point.
_CORE_ABI = {
    "get_config": (
        "vllm.transformers_utils.config",
        "88746e89a05b6c033dbef7be7253d9ed32eda373d0323ba417d742d4ffeb004e",
    ),
    "EngineArgs": (
        "vllm.engine.arg_utils",
        "003d76421f98b9df4b8d03a58d58903ba2aa91a8b7be436ab7888caae84d5aec",
    ),
    "HiSparseConfig": (
        "vllm.config.attention",
        "2aa8f28a28438b3564d6d15e65fd284b118637d7055ba3cabec0e30c61a7f6b0",
    ),
    "AttentionConfig": (
        "vllm.config.attention",
        "7c5b109a8af60c6082f48dcfc0efd63b475e7b010feccf9633ef3e1cd1ee4c91",
    ),
    "EngineArgs.__post_init__": (
        "vllm.engine.arg_utils",
        "c4f500452e03506c9e8d69e92354b031088d40ed117aa50ce37f03c9286e0a3e",
    ),
    "EngineArgs.create_model_config": (
        "vllm.engine.arg_utils",
        "89352a99b801a1f556635cb81c11fb38ed20278bbbf53ac8562c6d0986eb3f70",
    ),
    "EngineArgs.create_engine_config": (
        "vllm.engine.arg_utils",
        "e5286d15c01f4b4a301d428ace417b2c7cda44677a6dfce80180595bfb51051d",
    ),
    "VllmConfig.__post_init__": (
        "vllm.config.vllm",
        "b5365947fdc7aca763fb9f82bc69580648c4bb20cb48bd2c2d8c79d3521b480a",
    ),
    "VllmConfig.use_v2_model_runner": (
        "vllm.config.vllm",
        "f8a01833bc445227636c675ea14cc339eba642347f73c94651b79574769bac77",
    ),
    "VllmConfig": (
        "vllm.config.vllm",
        "aa5dd9c152e478bb4fffbd416b8d9a78e7116536174bdc1520ecc8dadf415daa",
    ),
}
_lock = RLock()
_state = None
_failure = None
_publishing = False
_constructed = {}


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if _state is not None or _publishing:
        _failure = "HiSparse config publication inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _source_abi(cls):
    module, expected = _CORE_ABI[cls.__qualname__]
    if cls.__module__ != module:
        raise RuntimeError("HiSparse reviewed Core class owner changed")
    source = textwrap.dedent(inspect.getsource(cls))
    digest = hashlib.sha256(
        ast.dump(ast.parse(source).body[0], include_attributes=False).encode()
    ).hexdigest()
    if digest != expected:
        raise RuntimeError(f"HiSparse reviewed Core source ABI changed: {cls.__name__}")


def _load_bindings():
    vllm = import_module("vllm")
    if vllm.__version__.split("+")[0] != "0.33.0":
        raise RuntimeError("HiSparse config compatibility requires pinned Core 0.33.0")
    args = import_module("vllm.engine.arg_utils")
    config = import_module("vllm.config.vllm")
    attention = import_module("vllm.config.attention")
    compilation = import_module("vllm.config.compilation")
    platforms = import_module("vllm.platforms")
    return (
        args,
        config,
        attention,
        compilation,
        platforms,
        import_module("vllm.envs"),
        platforms.current_platform,
        import_module("torch"),
        import_module("vllm.transformers_utils.config"),
    )


def _owned_health():
    if _failure is not None:
        raise RuntimeError(_failure)
    if _state is None:
        raise RuntimeError("HiSparse config compatibility has not been installed")
    if _state["pid"] != os.getpid():
        raise RuntimeError(
            "HiSparse config compatibility cannot be inherited across fork"
        )
    live = _load_bindings()
    if len(live) != len(_state["sources"]) or any(
        a is not b for a, b in zip(live, _state["sources"])
    ):
        raise RuntimeError("HiSparse Core/platform module bindings changed")
    if _is_hcu(live) != _state["hcu"]:
        raise RuntimeError("HiSparse canonical HCU platform owner changed")
    for module in live[:6] + live[7:]:
        if sys.modules.get(module.__name__) is not module:
            raise RuntimeError("HiSparse module identity changed")
    for owner, name, expected in _state["owners"]:
        if getattr(owner, name) is not expected:
            raise RuntimeError(f"HiSparse config owner changed: {name}")
    for function, expected in _state["callables"]:
        if _callable_state(function) != expected:
            raise RuntimeError("HiSparse callable code/signature/metadata drift")
    return {
        "installed": True,
        "pid": _state["pid"],
        "wrapped_points": ("create_engine_config",),
        "scope": "HCU Flash eager V2 BF16; explicit planner owner required for connector/DFly5 metadata",
        "runtime_admitted": False,
        "core_pin": "4ba787ec9959e566d3b587a796dff07673324da5",
    }


def check_health() -> dict:
    """Check process/ABI/publication ownership, not runtime qualification.

    Publication drift is latched; reinstalling cannot silently take over.
    DFly owns different entry points and can be installed before or after us.
    """
    global _failure
    with _lock:
        try:
            return _owned_health()
        except BaseException as error:
            if _state is not None and _failure is None:
                _failure = f"HiSparse config health failed: {error}; restart process"
            raise


def _callable_state(function):
    # Identity alone does not detect in-place __code__/defaults/signature edits.
    return (
        function.__code__,
        repr(inspect.signature(function, follow_wrapped=False)),
        deepcopy(function.__defaults__),
        deepcopy(function.__kwdefaults__),
        function.__module__,
        function.__qualname__,
        getattr(function, "__wrapped__", None),
    )


def _metadata_candidate(args, bindings):
    """Conservative selection only; the actual ModelConfig is authoritative.

    Reading HF metadata does not construct ModelConfig/VllmConfig or activate a
    platform. Unknown/failed metadata keeps *all* original factory semantics.
    No callable overrides or remote config code may be executed twice here.
    """
    overrides = getattr(args, "hf_overrides", {})
    if not isinstance(overrides, dict) or getattr(args, "trust_remote_code", False):
        return False
    try:
        hf = bindings[8].get_config(
            getattr(args, "hf_config_path", None) or args.model,
            False,
            getattr(args, "revision", None),
            getattr(args, "code_revision", None),
            getattr(args, "config_format", "auto"),
            hf_overrides_kw=deepcopy({
                key: value for key, value in overrides.items()
                if not isinstance(value, dict)
            }),
            token=getattr(args, "hf_token", None),
        )
        arches = getattr(hf, "architectures", None) or ()
        return bool(arches) and set(arches).issubset(_ARCHES) and getattr(
            hf, "model_type", None
        ) in ("hy_v4_flash", "hy_v4_flash_vl")
    except Exception:
        # Native validation remains authoritative; do not substitute our error.
        return False


def _is_hcu(bindings):
    platform_type = type(bindings[6])
    if platform_type.__module__ != "hcu_adapter.bridges.vllm":
        return False
    # A module-name lookalike is not admission. Never spoof CUDA or select a
    # platform here: normal adapter activation must already own this instance.
    return platform_type is import_module("hcu_adapter.bridges.vllm").HcuPlatform


def _is_flash(model):
    arches = getattr(model, "architectures", None) or ()
    text = getattr(model, "hf_text_config", None)
    return (
        bool(arches)
        and set(arches).issubset(_ARCHES)
        and getattr(text, "model_type", None) == "hy_v4_flash"
        and getattr(model.hf_config, "model_type", None)
        in ("hy_v4_flash", "hy_v4_flash_vl")
    )


def _require_model(model, bindings):
    if not _is_flash(model):
        raise ValueError(
            "HiSparse extension requires an actual HYV4 Flash text/VL model"
        )
    if getattr(model, "is_submodel_config", False) or getattr(
        model, "is_mm_encoder_only", False
    ):
        raise ValueError(
            "HiSparse requires a full language model, not an encoder/submodel view"
        )
    if bindings[5].VLLM_USE_V2_MODEL_RUNNER is not True:
        raise ValueError("HCU HiSparse requires explicit VLLM_USE_V2_MODEL_RUNNER=1")
    if not model.enforce_eager or model.dtype != bindings[7].bfloat16:
        raise ValueError(
            "HCU HiSparse requires explicit eager execution and BF16 model dtype"
        )
    # VL outer configs need not have index_topk. Core's text view is the DSA owner.
    topk = getattr(model.hf_text_config, "index_topk", None)
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError(
            "HiSparse requires a DSA hf_text_config with positive index_topk"
        )


def _planner_scope(value, *, final):
    # This opt-in is NEVER inferred from a generic KV-transfer object or spec.
    # Both real configs stay present throughout the single native construction.
    kv = value.kv_transfer_config
    if getattr(kv, "kv_connector", None) != "HCUHiSparseConnector":
        return False
    from . import hisparse_planner_compat
    hisparse_planner_compat.validate_config_scope(value, final=final)
    return True


def _planner_snapshot(config):
    kv = config.kv_transfer_config
    spec = config.speculative_config
    if not _planner_scope(config, final=True):
        return None
    # Identity is recorded separately. Snapshot non-model speculative fields as
    # values, and the actual draft HF metadata, rather than relying on a hash
    # which deliberately excludes cache capacity and host_pool_gib.
    return (
        deepcopy(vars(kv)),
        config.model_config.hf_text_config.to_dict(),
        (config.model_config.max_model_len, config.cache_config.block_size,
         config.scheduler_config.max_num_seqs, config.scheduler_config.max_num_batched_tokens,
         config.scheduler_config.async_scheduling, config.num_lookahead_tokens),
        None if spec is None else (
            type(spec),
            deepcopy({k: v for k, v in vars(spec).items()
                      if k not in ("target_model_config", "target_parallel_config",
                                   "draft_model_config", "draft_parallel_config")}),
            spec.draft_model_config.hf_config.to_dict(),
            (id(spec.draft_model_config), spec.draft_model_config),
            (id(spec.draft_parallel_config), spec.draft_parallel_config),
            spec.draft_model_config.dtype, spec.draft_model_config.enforce_eager,
        ),
    )


def _request_bounds(args, bindings):
    # Fail before ParallelConfig can discover/start distributed infrastructure.
    if any(getattr(args, name) != 1 for name in (
        "data_parallel_size", "pipeline_parallel_size",
        "prefill_context_parallel_size", "decode_context_parallel_size",
    )):
        raise ValueError("HCU HiSparse requires DP/PP/PCP/DCP=1")
    if getattr(args, "disable_hybrid_kv_cache_manager", None) is True:
        raise ValueError("HiSparse requires the hybrid KV cache manager (HMA)")
    if getattr(args, "scheduler_reserve_full_isl", None) is False:
        raise ValueError("HiSparse requires --scheduler-reserve-full-isl")
    if getattr(args, "kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
        raise ValueError("HCU HiSparse requires BF16 KV cache")
    planner_scope = _planner_scope(args, final=False)
    if (args.kv_transfer_config is not None and not planner_scope) or getattr(args, "kv_offloading_size", None) is not None:
        raise ValueError(
            "HCU HiSparse admits only the native host-pool path; no KV connector/offloading"
        )
    if not planner_scope and any(
        getattr(args, name, None) is not None
        for name in (
            "speculative_config",
            "spec_method",
            "spec_model",
            "spec_tokens",
        )
    ):
        raise NotImplementedError("HCU HiSparse + speculation/DFly remains unqualified")
    if getattr(args, "enable_lora", False) or getattr(args, "enable_dbo", False):
        raise ValueError("HCU HiSparse does not admit LoRA or DBO")
    if getattr(args, "enforce_eager", None) is not True:
        raise ValueError("HCU HiSparse requires explicit enforce_eager=True")
    compilation = args.compilation_config
    if compilation.cudagraph_mode not in (None, bindings[3].CUDAGraphMode.NONE):
        raise ValueError("HCU HiSparse is eager-only; cudagraph_mode must be NONE")
    if compilation.mode not in (None, bindings[3].CompilationMode.NONE):
        raise ValueError("HCU HiSparse requires compilation mode NONE")


def _require_hisparse(hisparse, bindings):
    if type(hisparse) is not bindings[2].HiSparseConfig:
        raise TypeError("HiSparse requires a real Core HiSparseConfig")
    size = hisparse.device_buffer_size
    if size is not None and (type(size) is not int or size <= 0):
        raise ValueError("HiSparse device_buffer_size must be a positive integer")
    if type(hisparse.eager_host_mirror) is not bool:
        raise ValueError("HiSparse eager_host_mirror must be a boolean")


def _validate_result(config, bindings):
    if not _is_hcu(bindings):
        raise ValueError("HiSparse extension is HCU-only")
    _require_model(config.model_config, bindings)
    if not config.use_v2_model_runner:
        raise ValueError("HiSparse requires Model Runner V2")
    p = config.parallel_config
    if any(
        getattr(p, name) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("HCU HiSparse requires DP/PP/PCP/DCP=1")
    if getattr(p, "enable_dbo", False):
        raise ValueError("HCU HiSparse does not admit DBO")
    c = config.compilation_config
    if (
        c.cudagraph_mode != bindings[3].CUDAGraphMode.NONE
        or c.mode != bindings[3].CompilationMode.NONE
    ):
        raise ValueError(
            "HCU HiSparse requires eager execution without compilation/graphs"
        )
    s = config.scheduler_config
    if not s.scheduler_reserve_full_isl:
        raise ValueError("HiSparse requires --scheduler-reserve-full-isl")
    if s.disable_hybrid_kv_cache_manager is not False:
        raise ValueError("HiSparse requires the hybrid KV cache manager (HMA)")
    planner_scope = _planner_scope(config, final=True)
    if (config.kv_transfer_config is not None and not planner_scope) or getattr(
        config.cache_config, "kv_offloading_size", None
    ) is not None:
        raise ValueError(
            "HCU HiSparse admits only the native host-pool path; no KV connector/offloading"
        )
    if config.speculative_config is not None and not planner_scope:
        raise NotImplementedError("HCU HiSparse + speculation/DFly remains unqualified")
    if config.lora_config is not None:
        raise ValueError("HCU HiSparse does not admit LoRA")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("HCU HiSparse requires BF16 KV cache")
    attention = config.attention_config
    if (
        getattr(attention.backend, "name", None) != "FLASHMLA_SPARSE"
        or attention.backend_per_kind
    ):
        raise ValueError(
            "HCU HiSparse requires explicit FLASHMLA_SPARSE without per-kind overrides"
        )
    _require_hisparse(attention.hisparse_config, bindings)


def _record(config):
    key = id(config)
    attention = config.attention_config
    hisparse = attention.hisparse_config

    def discard(reference):
        with _lock:
            entry = _constructed.get(key)
            if entry is not None and entry[0] is reference:
                _constructed.pop(key, None)

    # A weak record avoids extending the life of a model/config. Keep neither
    # a strong config ref nor a hashability requirement on Core's dataclass.
    _constructed[key] = (
        weakref.ref(config, discard),
        attention,
        hisparse,
        deepcopy(hisparse),
        config.model_config.hf_text_config,
        config.model_config.hf_text_config.index_topk,
        tuple((name, getattr(config, name)) for name in (
            "model_config", "parallel_config", "compilation_config",
            "scheduler_config", "cache_config", "speculative_config",
            "lora_config", "kv_transfer_config",
        )),
        _planner_snapshot(config),
    )


def validate_hisparse_config(vllm_config) -> None:
    """Non-HCU/non-Flash/non-HiSparse is inert; admitted configs need provenance.

    Rechecks final Core HMA/runner/DSA state and supported implementation bounds.
    It does not enable kernels, allocate host/GPU pools, or qualify DFly jointly.
    """
    with _lock:
        entry = _constructed.get(id(vllm_config))
        if entry is None and getattr(
            vllm_config.attention_config, "hisparse_config", None
        ) is None:
            return
        check_health()
        if not _is_hcu(_state["sources"]):
            return
        if entry is None and not _is_flash(vllm_config.model_config):
            return
        attention = vllm_config.attention_config
        if (
            entry is None
            or entry[0]() is not vllm_config
            or entry[1] is not attention
            or entry[2] is not attention.hisparse_config
            or entry[3] != attention.hisparse_config
            or entry[4] is not vllm_config.model_config.hf_text_config
            or entry[5] != vllm_config.model_config.hf_text_config.index_topk
            or any(getattr(vllm_config, name) is not owner for name, owner in entry[6])
            or entry[7] != _planner_snapshot(vllm_config)
        ):
            raise ValueError(
                "HiSparse config lacks intact owned EngineArgs construction provenance"
            )
        _validate_result(vllm_config, _state["sources"])


def _validate_planner_config(vllm_config):
    # Strict handshake: the ordinary public validator is intentionally inert on
    # non-targets, so it alone cannot certify a resolver's target config owner.
    with _lock:
        entry = _constructed.get(id(vllm_config))
        if entry is None or entry[0]() is not vllm_config:
            raise ValueError("HiSparse planner requires owned EngineArgs construction provenance")
        validate_hisparse_config(vllm_config)
        if not _planner_scope(vllm_config, final=True):
            raise ValueError("HiSparse planner requires its explicit connector scope")


def validate_hisparse_runtime(vllm_config) -> None:
    """Fail closed at the unqualified host-pool boundary after config validation.

    Non-HCU/non-Flash/non-HiSparse is inert. This Core pin unconditionally asks
    HiSparseConnector for host_pool_gib during KV planning. We do not synthesize
    it, change Core, or claim that a no-connector config can start an engine.
    Main must independently qualify a connector/planner extension before this
    gate can admit runtime use, and separately qualify HiSparse + DFly.
    """
    validate_hisparse_config(vllm_config)
    if (
        getattr(vllm_config.attention_config, "hisparse_config", None) is not None
        and _is_hcu(_state["sources"])
        and _is_flash(vllm_config.model_config)
    ):
        if _planner_scope(vllm_config, final=True):
            raise NotImplementedError(
                "HCU HiSparse planner metadata only: native connector scheduler/worker, "
                "HIP host mapping/transfer and HiSparse + DFly joint runtime remain unqualified"
            )
        raise NotImplementedError(
            "HCU HiSparse config-only: Core033 get_hisparse_host_pool_bytes "
            "requires HiSparseConnector with host_pool_gib; the requested "
            "no-connector engine path and HiSparse + DFly remain unqualified"
        )


def _copy_args(args):
    # EngineArgs is already normalized. deepcopy copies the complete instance
    # graph and its internal aliases WITHOUT re-running __init__/__post_init__.
    # dataclasses.replace(args) would repeat plugin loading, quantization config
    # resolution and offline path normalization; a shallow copy would expose
    # nested reasoning/spec/config dicts to native factory mutation.
    clone = deepcopy(args)
    if clone is args or type(clone) is not type(args):
        raise ValueError("HiSparse requires an independently copied EngineArgs")
    return clone


def _construct(original, args, usage_context, headless, bindings):
    if type(args).__post_init__ is not bindings[0].EngineArgs.__post_init__:
        raise ValueError("HiSparse does not admit an overridden EngineArgs post-init")
    clone = _copy_args(args)
    requested_hisparse = args.attention_config.hisparse_config
    _require_hisparse(requested_hisparse, bindings)
    # Revalidate all fields without modifying or replacing the caller's object.
    validated = replace(requested_hisparse)
    snapshot = deepcopy(requested_hisparse)
    if validated != snapshot:
        raise ValueError("HiSparse validation would normalize caller input")
    if "create_model_config" in vars(args):
        raise ValueError(
            "HiSparse requires the unmodified Core EngineArgs model factory"
        )
    model_factory = clone.create_model_config
    if (
        getattr(model_factory, "__func__", None)
        is not bindings[0].EngineArgs.create_model_config
    ):
        raise ValueError("HiSparse does not admit an overridden model config factory")
    captured = []

    def create_model_config():
        if captured:
            raise RuntimeError("Reviewed Core must construct ModelConfig exactly once")
        model = model_factory()
        admitted = _is_flash(model)
        captured.append((model, admitted))
        if not admitted:
            raise ValueError("HiSparse metadata candidate disagrees with actual Core model")
        if admitted:
            _require_model(model, bindings)
            _request_bounds(clone, bindings)
            clone.attention_config = replace(
                clone.attention_config, hisparse_config=None
            )
        return model

    # This is a private instance hook, not a class/Core/global publication. It
    # sees the real resolved ModelConfig without loading HF config twice.
    clone.create_model_config = create_model_config
    try:
        config = original(clone, usage_context=usage_context, headless=headless)
    finally:
        del clone.create_model_config
    if len(captured) != 1:
        raise RuntimeError("Reviewed Core did not construct exactly one ModelConfig")
    model, _ = captured[0]
    if (
        config.model_config is not model
        or config.attention_config.hisparse_config is not None
    ):
        raise RuntimeError(
            "HiSparse Core factory output identity/attention state drift"
        )
    if args.attention_config.hisparse_config is not requested_hisparse or requested_hisparse != snapshot:
        raise ValueError("HiSparse caller input mutated during Core construction")
    config.attention_config = replace(
        config.attention_config, hisparse_config=requested_hisparse
    )
    if config.attention_config.hisparse_config is not requested_hisparse:
        raise RuntimeError("Core AttentionConfig changed the real HiSparseConfig identity")
    _validate_result(config, bindings)
    with _lock:
        check_health()
        _record(config)
    validate_hisparse_config(config)
    return config


def install_compat() -> bool:
    """Explicitly publish the reviewed wrapper; repeated healthy install is inert.

    Returns True on first publication, False if already installed. A partial
    publication/fork/drift requires process restart. Source preflight failure
    before publication remains retryable and leaves Core untouched.
    """
    global _state, _failure, _publishing
    with _lock:
        if _failure is not None:
            raise RuntimeError(_failure)
        if _state is not None:
            check_health()
            return False
        if _publishing:
            _failure = "recursive HiSparse compatibility publication; restart process"
            raise RuntimeError(_failure)
        _publishing = True
        changed = False
        original = None
        args_cls = None
        wrapper = None
        try:
            bindings = _load_bindings()
            args_module, config_module = bindings[:2]
            args_cls = args_module.EngineArgs
            config_cls = config_module.VllmConfig
            _source_abi(args_cls)
            _source_abi(config_cls)
            _source_abi(bindings[2].AttentionConfig)
            _source_abi(bindings[2].HiSparseConfig)
            _source_abi(bindings[8].get_config)
            metadata = bindings[8].get_config
            if (
                tuple(inspect.signature(metadata).parameters) != (
                    "model", "trust_remote_code", "revision", "code_revision",
                    "config_format", "hf_overrides_kw", "hf_overrides_fn", "kwargs",
                )
                or hasattr(metadata, "__wrapped__")
            ):
                raise RuntimeError("HiSparse Core metadata factory signature/owner drift")
            if (
                args_module.VllmConfig is not config_cls
                or args_module.AttentionConfig is not bindings[2].AttentionConfig
                or args_module.current_platform is not bindings[6]
                or config_module.current_platform is not bindings[6]
            ):
                raise RuntimeError("HiSparse Core factory/platform alias has another owner")
            for method in (
                args_cls.create_engine_config,
                args_cls.create_model_config,
                args_cls.__post_init__,
                config_cls.__post_init__,
                config_cls.use_v2_model_runner.fget,
            ):
                _source_abi(method)
            original = args_cls.create_engine_config
            parameters = inspect.signature(original).parameters
            if (
                tuple(parameters) != ("self", "usage_context", "headless")
                or parameters["usage_context"].default is not None
                or parameters["headless"].default is not False
                or hasattr(original, "__wrapped__")
                or original.__module__ != "vllm.engine.arg_utils"
            ):
                raise RuntimeError(
                    "HiSparse Core033 public factory signature/owner drift"
                )
            for name in ("create_model_config", "__post_init__"):
                method = getattr(args_cls, name)
                if (
                    tuple(inspect.signature(method).parameters) != ("self",)
                    or hasattr(method, "__wrapped__")
                    or method.__module__ != "vllm.engine.arg_utils"
                ):
                    raise RuntimeError(f"HiSparse Core033 model factory drift: {name}")

            @wraps(original)
            def wrapper(self, usage_context=None, headless=False):
                check_health()
                attention = getattr(self, "attention_config", None)
                if (
                    not _is_hcu(bindings)
                    or getattr(attention, "hisparse_config", None) is None
                    or not _metadata_candidate(self, bindings)
                ):
                    return original(
                        self, usage_context=usage_context, headless=headless
                    )
                return _construct(original, self, usage_context, headless, bindings)

            owners = (
                (args_module, "EngineArgs", args_cls),
                (config_module, "VllmConfig", config_cls),
                (args_module, "VllmConfig", config_cls),
                (args_module, "AttentionConfig", bindings[2].AttentionConfig),
                (args_module, "current_platform", bindings[6]),
                (config_module, "current_platform", bindings[6]),
                (bindings[8], "get_config", metadata),
                (bindings[2], "AttentionConfig", bindings[2].AttentionConfig),
                (bindings[2], "HiSparseConfig", bindings[2].HiSparseConfig),
                (bindings[3], "CompilationMode", bindings[3].CompilationMode),
                (bindings[3], "CUDAGraphMode", bindings[3].CUDAGraphMode),
                (args_cls, "create_engine_config", wrapper),
                (args_cls, "create_model_config", args_cls.create_model_config),
                (args_cls, "__post_init__", args_cls.__post_init__),
                (config_cls, "__post_init__", config_cls.__post_init__),
                (config_cls, "use_v2_model_runner", config_cls.use_v2_model_runner),
                (bindings[4], "current_platform", bindings[6]),
            )
            args_cls.create_engine_config = wrapper
            changed = True
            _state = {
                "pid": os.getpid(),
                "sources": bindings,
                "owners": owners,
                "hcu": _is_hcu(bindings),
                "callables": tuple(
                    (fn, _callable_state(fn)) for fn in (
                        original, wrapper, args_cls.create_model_config,
                        args_cls.__post_init__, config_cls.__post_init__,
                        config_cls.use_v2_model_runner.fget, metadata,
                    )
                ),
            }
            check_health()
            return True
        except BaseException as error:
            if changed:
                _failure = f"HiSparse config partial publication failed: {error}; restart process"
                if args_cls.create_engine_config is wrapper:
                    args_cls.create_engine_config = original
                _state = None
            raise
        finally:
            _publishing = False
