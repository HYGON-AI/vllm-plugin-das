# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact HY4 W4A8 checkpoint admission, independent of runtime/provider imports."""
from .ct_w4a8_schema import schema_matches

NAME = 'hcu_hy4_ct_w4a8'


def is_hy4(config):
    return getattr(config, 'model_type', None) == 'hy_v4'


def accepts_checkpoint(quant, user_quant, hf_config):
    if not is_hy4(hf_config) or user_quant not in (None, 'compressed-tensors', NAME):
        return False
    if not isinstance(quant, dict) or quant.get('quant_method') != 'compressed-tensors':
        return False
    groups = quant.get('config_groups')
    if not isinstance(groups, dict):
        return False
    return any(isinstance(group, dict)
               and group.get('format', quant.get('format')) == 'pack-quantized'
               and schema_matches(group.get('weights'), group.get('input_activations'))
               for group in groups.values())
