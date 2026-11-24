# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Lazy native HY4 model composition for explicitly requested CT admission."""
from hcu_adapter.extensions import ExtensionCommitError

_MODULE = 'hcu_vllm_das.models.hy4_ct'
_MODELS = {'HYV4ForCausalLM': ('HYV4ForCausalLM', 'HcuHY4CTForCausalLM'),
           'HYV4MTPModel': ('HYV4MTP', 'HcuHY4CTMTP')}
_publication = None


def _registry():
    from vllm.model_executor.models.registry import ModelRegistry
    return ModelRegistry


def _key(entry):
    return getattr(entry, 'module_name', None), getattr(entry, 'class_name', None)


def check_health():
    if _publication is None:
        return
    registry, register, entries = _publication
    if (registry.register_model != register
            or any(registry.models.get(arch) is not entries[arch]
                   or _key(entries[arch]) != (_MODULE, owned)
                   for arch, (_, owned) in _MODELS.items())):
        raise ExtensionCommitError('HY4 CT model registry changed owner')


def activate(context):
    global _publication
    # Static composition includes the same format loader and publishes its own
    # models at the next extension priority. It must keep sole registry ownership.
    if ('hy4-static-eplb' in context.requested_capabilities
            or 'das_hcu_hy4_static' in context.requested_extensions
            or not ('hy4-ct-w4a8' in context.requested_capabilities
                    or 'das_hcu_hy4_ct_w4a8' in context.requested_extensions)):
        return
    if _publication is not None:
        check_health()
        return
    registry = _registry()
    for arch, (native, _) in _MODELS.items():
        if _key(registry.models.get(arch)) != ('vllm.models.hy_v4', native):
            raise RuntimeError('HY4 CT model has a foreign owner: ' + arch)
    register = registry.register_model
    try:
        for arch, (_, owned) in _MODELS.items():
            register(arch, _MODULE + ':' + owned)
        _publication = registry, register, {arch: registry.models[arch] for arch in _MODELS}
        check_health()
    except BaseException as error:
        raise ExtensionCommitError('HY4 CT model publication failed') from error
