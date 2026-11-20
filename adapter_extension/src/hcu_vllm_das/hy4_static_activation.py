# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional static HY4 model composition through Core's public registry."""
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

_ENTRY = 'das_hcu_hy4_static'
_CAPABILITY = 'hy4-static-eplb'
_MODULE = 'hcu_vllm_das.models.hy4_static'
_MODELS = {
    'HYV4ForCausalLM': ('HYV4ForCausalLM', 'HcuHY4StaticForCausalLM'),
    'HYV4MTPModel': ('HYV4MTP', 'HcuHY4StaticMTP'),
}
_publication = None


def _registry():
    from vllm.model_executor.models.registry import ModelRegistry
    return ModelRegistry


def _key(entry):
    return getattr(entry, 'module_name', None), getattr(entry, 'class_name', None)


def _function(function):
    return getattr(function, '__func__', function)


def check_health():
    if _publication is None:
        raise RuntimeError('HY4 static model publication is not active')
    registry, register, entries = _publication
    if (_function(registry.register_model) is not register
            or any(registry.models.get(arch) is not entries[arch]
                   or _key(entries[arch]) != (_MODULE, owned)
                   for arch, (_, owned) in _MODELS.items())):
        raise ExtensionCommitError('HY4 static model registry changed owner')


def activate(context):
    global _publication
    if (_CAPABILITY not in context.requested_capabilities
            and _ENTRY not in context.requested_extensions):
        raise RuntimeError('HY4 static EPLB requires an explicit capability/extension request')
    if _publication is not None:
        check_health()
        return
    from .hy4_quant_activation import check_health as check_quantization
    check_quantization()
    registry = _registry()
    # Preflight BOTH names before changing either. Discovery remains free of
    # Core/Torch; actual model classes load only through the public lazy API.
    for arch, (native, _) in _MODELS.items():
        if _key(registry.models.get(arch)) != ('vllm.models.hy_v4', native):
            raise RuntimeError('HY4 static model has a foreign owner: ' + arch)
    register = _function(registry.register_model)
    try:
        for arch, (_, owned) in _MODELS.items():
            registry.register_model(arch, _MODULE + ':' + owned)
        entries = {arch: registry.models[arch] for arch in _MODELS}
        _publication = (registry, register, entries)
        check_health()
    except BaseException as error:
        # Core has no model-unregister operation. Adapter owns serialization,
        # current-process health and the persistent latch for this public signal.
        raise ExtensionCommitError('HY4 static model publication failed') from error


def extension_spec():
    return ExtensionSpec(name='das-hcu-hy4-static-eplb', api_version=1,
        priority=103, capabilities=frozenset({_CAPABILITY}),
        activate=activate, check_health=check_health)
