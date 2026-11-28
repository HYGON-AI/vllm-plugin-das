# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 VL publication through the existing Adapter lifecycle."""

import os
from importlib import import_module
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from packaging.version import InvalidVersion, Version

from .model_runtime_activation import ModelRuntimeActivation

CAPABILITY = "hy4-vl"
_CONFIG = "hcu_vllm_das.models.hy4_standard.configs"
_MODEL = "hcu_vllm_das.models.hy4_standard.vl_model"
_ARCH = "HYV4VLForConditionalGeneration"
_TYPES = (("hy_v4_vl", "HYV4VLConfig"), ("hy_v4_vl_vision", "HYV4VLVisionConfig"))
_lock = RLock()
_metadata = None
_runtime = None
_failure = None
_publishing = False


def _fatal(reason):
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if (_runtime is not None or _publishing) and _failure is None:
        _failure = "standard HY4 VL runtime inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _check_metadata():
    if _metadata is None:
        raise RuntimeError("standard HY4 VL metadata has not activated")
    pid, mapping, classes, text_class = _metadata
    module = import_module(_CONFIG)
    core_text = import_module("vllm.transformers_utils.configs.hy_v4").HYV4Config
    if (
        import_module("transformers").CONFIG_MAPPING is not mapping
        or core_text is not text_class
        or core_text.model_type != "hy_v4"
        or any(
            mapping[key] is not cls or getattr(module, symbol) is not cls
            for (key, symbol), cls in zip(_TYPES, classes)
        )
    ):
        _fatal("standard HY4 VL config ownership changed")
    if pid != os.getpid():
        raise RuntimeError("standard HY4 VL config requires activation after fork")


def _check_runtime():
    if _runtime is None:
        raise RuntimeError("standard HY4 VL runtime has not activated")
    (
        pid,
        registry,
        entry,
        renderers,
        renderer_entry,
        tokenizer,
        tokenizer_entry,
        hf_methods,
        owned_module,
        owned_class,
        model_module,
        model_class,
    ) = _runtime
    hf = import_module("vllm.renderers.hf").HfRenderer
    if (
        import_module("vllm.model_executor.models").ModelRegistry is not registry
        or registry.models.get(_ARCH) is not entry
        or import_module("vllm.renderers.registry").RENDERER_REGISTRY is not renderers
        or renderers.renderers.get("hf") != renderer_entry
        or import_module("vllm.tokenizers.registry").TokenizerRegistry is not tokenizer
        or tokenizer.tokenizers.get("hf") != tokenizer_entry
        or (hf.render_messages, hf.render_messages_async) != hf_methods
        or import_module(owned_module.__name__) is not owned_module
        or getattr(owned_module, "HY4VLHfRenderer", None) is not owned_class
        or import_module(model_module.__name__) is not model_module
        or getattr(model_module, _ARCH, None) is not model_class
        or (getattr(entry, "module_name", None), getattr(entry, "class_name", None))
        != (_MODEL, _ARCH)
    ):
        _fatal("standard HY4 VL runtime registry/stock HF ownership changed")
    if pid != os.getpid():
        _fatal("standard HY4 VL runtime requires a fresh spawn process")


def check_health():
    """Read-only owner checks; publication never repairs foreign registrations."""
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        try:
            _check_metadata()
            if _runtime is not None:
                _check_runtime()
        except ExtensionCommitError:
            raise
        except (KeyError, AttributeError) as error:
            _fatal(f"standard HY4 VL publication unavailable: {error}")


def _publish_runtime(context):
    global _runtime, _publishing, _failure
    with _lock:
        check_health()
        if _runtime is not None:
            _check_runtime()
            return
        if _publishing:
            _fatal("recursive standard HY4 VL runtime publication")
        registry = import_module("vllm.model_executor.models").ModelRegistry
        renderers = import_module("vllm.renderers.registry").RENDERER_REGISTRY
        tokenizer = import_module("vllm.tokenizers.registry").TokenizerRegistry
        renderer_entry = renderers.renderers.get("hf")
        tokenizer_entry = tokenizer.tokenizers.get("hf")
        if _ARCH in registry.models:
            raise RuntimeError("standard HY4 VL model already has a registry owner")
        if renderer_entry != ("vllm.renderers.hf", "HfRenderer"):
            raise RuntimeError("standard HY4 VL requires the stock HF renderer owner")
        if tokenizer_entry != ("vllm.tokenizers.hf", "CachedHfTokenizer"):
            raise RuntimeError("standard HY4 VL requires the stock cached HF tokenizer")
        hf = import_module("vllm.renderers.hf").HfRenderer
        owned_module = import_module("hcu_vllm_das.models.hy4_standard.renderer")
        owned = owned_module.HY4VLHfRenderer
        if not issubclass(owned, hf):
            raise TypeError("standard HY4 VL renderer has an incompatible base")
        _publishing = True
        try:
            model_module = import_module(_MODEL)
            model_class = getattr(model_module, _ARCH)
            registry.register_model(_ARCH, f"{_MODEL}:{_ARCH}")
            renderers.register("hf", owned.__module__, owned.__name__)
            _runtime = (
                os.getpid(),
                registry,
                registry.models[_ARCH],
                renderers,
                renderers.renderers["hf"],
                tokenizer,
                tokenizer_entry,
                (hf.render_messages, hf.render_messages_async),
                owned_module,
                owned,
                model_module,
                model_class,
            )
            _check_runtime()
        except BaseException as error:
            _failure = "standard HY4 VL runtime commit failed; restart a clean process"
            raise ExtensionCommitError(_failure) from error
        finally:
            _publishing = False


_model_activation = ModelRuntimeActivation(_publish_runtime, spawn_by_default=True)


def request_model_runtime():
    _model_activation.request()


def activate(context):
    global _metadata, _publishing, _failure
    try:
        version = Version(context.vllm_version or "")
    except InvalidVersion:
        version = None
    if version is None or version.epoch or version.release[:3] != (0, 33, 0):
        raise RuntimeError("standard HY4 VL requires stock Core 0.33.0")
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _metadata is None:
            transformers = import_module("transformers")
            mapping = transformers.CONFIG_MAPPING
            if any(key in mapping for key, _ in _TYPES):
                raise RuntimeError(
                    "standard HY4 VL config already has a registry owner"
                )
            module = import_module(_CONFIG)
            classes = tuple(getattr(module, symbol) for _, symbol in _TYPES)
            text_class = import_module(
                "vllm.transformers_utils.configs.hy_v4"
            ).HYV4Config
            _publishing = True
            try:
                for (key, _), cls in zip(_TYPES, classes):
                    transformers.AutoConfig.register(key, cls)
                _metadata = (os.getpid(), mapping, classes, text_class)
                _check_metadata()
            except BaseException as error:
                _failure = (
                    "standard HY4 VL config commit failed; restart a clean process"
                )
                raise ExtensionCommitError(_failure) from error
            finally:
                _publishing = False
        else:
            if _metadata[0] != os.getpid():
                _metadata = (os.getpid(), *_metadata[1:])
            _check_metadata()
    _model_activation.bind(context)
    check_health()


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-hy4-vl",
        api_version=1,
        priority=102,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
