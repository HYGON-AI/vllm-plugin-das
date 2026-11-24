# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 static placement scope before physical parameter allocation."""
from types import SimpleNamespace as NS

import pytest

from hcu_vllm_das.hy4_static_scope import applies, owns_layer, validate_scope


def config():
    return NS(model_config=NS(enforce_eager=True,
        hf_config=NS(model_type='hy_v4', num_hidden_layers=78,
                     num_nextn_predict_layers=1)),
        lora_config=None,
        attention_config=NS(backend=NS(name='FLASHMLA_SPARSE')),
        parallel_config=NS(prefill_context_parallel_size=16,
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
            decode_context_parallel_size=1, eplb_config=NS(num_redundant_experts=0)))


def options():
    return NS(recording=False, num_redundant_experts=16)


def test_exact_scope_and_both_backbone_and_mtp_layer_ids():
    cfg = config()
    assert applies(cfg)
    assert validate_scope(cfg, options())
    for prefix in ('model.layers.3.mlp', 'model.layers.77.mlp',
                   'model.layers.78.mtp_block.mlp',
                   'model.layers.78.mlp.experts'):
        assert owns_layer(cfg, prefix)
    assert not owns_layer(cfg, 'model.layers.79.mlp')
    cfg.model_config.hf_config.model_type = 'hy_v4_mtp'
    assert applies(cfg)
    assert owns_layer(cfg, 'model.layers.78.mtp_block.mlp')
    cfg.model_config.hf_config.model_type = 'deepseek_v41'
    assert not applies(cfg)


@pytest.mark.parametrize('field,value', [('prefill_context_parallel_size', 8),
    ('tensor_parallel_size', 2), ('data_parallel_size', 2),
    ('pipeline_parallel_size', 2), ('decode_context_parallel_size', 2)])
def test_unsupported_topology_is_rejected(field, value):
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError, match='HY4 static'):
        validate_scope(cfg, options())


@pytest.mark.parametrize('change', ['graph', 'lora', 'backend', 'record',
                                   'global_redundancy', 'redundancy'])
def test_unreviewed_execution_and_provenance_modes_are_rejected(change):
    cfg, opt = config(), options()
    if change == 'graph': cfg.model_config.enforce_eager = False
    if change == 'lora': cfg.lora_config = object()
    if change == 'backend': cfg.attention_config.backend = None
    if change == 'record': opt.recording = True
    if change == 'global_redundancy': cfg.parallel_config.eplb_config.num_redundant_experts = 16
    if change == 'redundancy': opt.num_redundant_experts = True
    with pytest.raises(ValueError, match='HY4 static'):
        validate_scope(cfg, opt)


def test_uniform_simulation_is_not_reported_as_real_static_routing():
    with pytest.raises(ValueError, match='simulation'):
        validate_scope(config(), options(), routing_simulation='uniform_random')
