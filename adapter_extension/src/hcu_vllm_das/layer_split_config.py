# SPDX-License-Identifier: Apache-2.0
"""Process-independent Layer Split startup option."""

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


def register_cli():
    from hcu_adapter.bridges.vllm_cli import register_config_switch

    register_config_switch(FLAG, KEY,
                           help='Shard DeepSeek V4.1 KV cache layers across PCP ranks.',
                           validate=_validate_config)
