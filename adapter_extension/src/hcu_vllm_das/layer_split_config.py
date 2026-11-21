# SPDX-License-Identifier: Apache-2.0
"""Process-independent Layer Split startup option."""

from .adapter_api import optional_adapter_module

KEY = 'hcu_layer_split'
FLAG = '--enable-layer-split'


def layer_split_enabled(config):
    additional = getattr(config, 'additional_config', None)
    if not isinstance(additional, dict) or KEY not in additional:
        return False
    options = additional[KEY]
    if (not isinstance(options, dict) or options.keys() != {'enabled'}
            or type(options['enabled']) is not bool):
        raise ValueError('hcu_layer_split requires exactly one boolean field: enabled')
    return options['enabled']


def _validate_config(config):
    if layer_split_enabled(config):
        from .v41_pcp import validate_layer_split_policy
        from .dsa_cache_layer_split import split_config

        validate_layer_split_policy(config)
        if split_config(config) is None:
            raise ValueError('Layer Split requires PCP>1 and KV source layers')


def adapter_hooks():
    cli = optional_adapter_module('hcu_adapter.bridges.vllm_cli')
    layout = optional_adapter_module('hcu_adapter.bridges.vllm_kv_cache')
    return (cli, layout) if cli is not None and layout is not None else None


def register_cli():
    hooks = adapter_hooks()
    if hooks is None:
        return None
    cli, layout = hooks
    cli.register_config_switch(FLAG, KEY,
                               help='Shard DeepSeek V4.1 KV cache layers across PCP ranks.',
                               validate=_validate_config)
    return layout
