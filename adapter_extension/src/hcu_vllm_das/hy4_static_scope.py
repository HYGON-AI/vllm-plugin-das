# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exact HY4 static-placement scope, independent of host lifecycle imports."""
import re

from .quantization.hy4_ct_policy import is_hy4

_LAYER = re.compile(r'(?:^|\.)layers\.(\d+)\.(?:mtp_block\.)?mlp(?:\.experts)?$')


def applies(cfg):
    model = getattr(cfg, 'model_config', None)
    return is_hy4(getattr(model, 'hf_config', None))


def owns_layer(cfg, prefix):
    if not applies(cfg) or not isinstance(prefix, str):
        return False
    match = _LAYER.search(prefix)
    hf = cfg.model_config.hf_config
    depth = hf.num_hidden_layers
    draft = getattr(hf, 'num_nextn_predict_layers', 0)
    if (match is None or type(depth) is not int or depth < 1
            or type(draft) is not int or draft < 0):
        return False
    return int(match.group(1)) < depth + draft


def validate_scope(cfg, options, *, routing_simulation=''):
    errors = []
    if not applies(cfg):
        errors.append('native HY4 backbone/MTP configuration required')
    p = cfg.parallel_config
    for name, expected in (('prefill_context_parallel_size', 16),
            ('tensor_parallel_size', 1), ('data_parallel_size', 1),
            ('pipeline_parallel_size', 1), ('decode_context_parallel_size', 1)):
        value = getattr(p, name)
        if type(value) is not int or value != expected:
            errors.append(f'{name} must be {expected}')
    if not cfg.model_config.enforce_eager:
        errors.append('eager execution required for the initial PCP16 scope')
    if cfg.lora_config is not None:
        errors.append('LoRA is outside the static layout contract')
    backend = cfg.attention_config.backend
    if getattr(backend, 'name', backend) != 'FLASHMLA_SPARSE':
        errors.append('FLASHMLA_SPARSE required to retain the HY4 sink')
    if options.recording:
        errors.append('HY4 calibration is not implemented in this deployment scope')
    if type(options.num_redundant_experts) is not int or options.num_redundant_experts < 0:
        errors.append('redundancy must be a nonnegative integer')
    if p.eplb_config.num_redundant_experts != 0:
        errors.append('redundant slots belong in hcu_static_eplb, not Core eplb_config')
    if routing_simulation:
        errors.append('routing simulation cannot be combined with real static placement')
    if errors:
        raise ValueError('HY4 static scope: ' + '; '.join(errors))
    return True
