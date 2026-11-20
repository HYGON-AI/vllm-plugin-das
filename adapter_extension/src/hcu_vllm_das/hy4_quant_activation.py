# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 provider health; the Adapter owns the CT registration transaction."""
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .adapter_api import adapter_module
from .quantization.hy4_ct_policy import NAME


def check_health():
    owner = adapter_module('ct_w4a8_registration')
    owner.check_health()
    from vllm.model_executor.layers.quantization import get_quantization_config
    from .quantization.hy4_ct_w4a8 import HcuHY4CTW4A8Config
    if get_quantization_config(NAME) is not HcuHY4CTW4A8Config:
        raise ExtensionCommitError('HY4 W4A8 config registration changed owner')
    from .hy4_ct_model_activation import check_health as check_models
    check_models()


def activate(context):
    from ._seams import AdapterContractError
    owner = adapter_module('ct_w4a8_registration')
    if not callable(getattr(owner, 'set_additional_config_provider', None)):
        raise AdapterContractError(
            'HY4 W4A8 requires Adapter CT set_additional_config_provider')
    check_health()
    from .hy4_ct_model_activation import activate as activate_models
    activate_models(context)


def extension_spec():
    # Adapter dispatches larger priorities first. CT publishes at 105; this
    # capability verifies that transaction only after it has completed.
    return ExtensionSpec(name='das-hcu-hy4-ct-w4a8', api_version=1, priority=104,
        capabilities=frozenset({'hy4-ct-w4a8'}), activate=activate,
        check_health=check_health)
