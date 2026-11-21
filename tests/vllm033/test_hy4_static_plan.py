# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Legacy HY4 map admission, using real CPU tensors only."""
import hashlib
import json

import pytest

from hcu_vllm_das.hy4_static_plan import load_hy4_plan
from hcu_vllm_das.static_eplb import load_static_eplb_plan


def payload():
    row = [2, 0, 3, 1, 0, 1]
    maps = {}
    for key, layers in [('HYV4ForCausalLM', 2), ('HYV4MTP', 1)]:
        maps[key] = dict(model_class=key, num_logical_experts=4,
            num_moe_layers=layers, num_physical_experts=6,
            num_redundant_experts=2,
            physical_to_logical_map=[list(row) for _ in range(layers)])
    return dict(version=2, model_maps=maps)


def read(path, key='HYV4ForCausalLM'):
    return load_hy4_plan(path, model_key=key, expected_model_name='test-hy4',
        expected_shape=(2 if key == 'HYV4ForCausalLM' else 1, 6),
        num_logical_experts=4, num_redundant_experts=2)


@pytest.mark.parametrize('key', ['HYV4ForCausalLM', 'HYV4MTP'])
def test_native_map_keys_counts_and_source_provenance(tmp_path, key):
    path = tmp_path / 'r16.json'
    raw = json.dumps(payload()).encode()
    path.write_bytes(raw)
    plan = read(path, key)
    assert plan.model_key == key
    assert plan.num_physical_experts == 6
    assert plan.layer_map(0) == (2, 0, 3, 1, 0, 1)
    assert plan.source_sha256 == hashlib.sha256(raw).hexdigest()
    # The legacy artifact records no calibration kind or checkpoint binding.
    assert plan.record_kind == 'initial'
    copied = plan.physical_to_logical_map
    copied.fill_(-1)
    assert plan.layer_map(0) == (2, 0, 3, 1, 0, 1)


@pytest.mark.parametrize('change', ['version', 'format', 'model_key', 'bool_count',
    'physical_count', 'layer_count', 'missing_expert', 'invalid_expert',
    'bool_expert', 'row_width', 'record_kind'])
def test_bad_legacy_metadata_or_slots_are_rejected(tmp_path, change):
    doc = payload()
    model = doc['model_maps']['HYV4ForCausalLM']
    if change == 'version': doc['version'] = True
    if change == 'format': doc['format'] = 'foreign-format'
    if change == 'model_key': model['model_class'] = 'OtherModel'
    if change == 'bool_count': model['num_redundant_experts'] = True
    if change == 'physical_count': model['num_physical_experts'] = 7
    if change == 'layer_count': model['physical_to_logical_map'].pop()
    if change == 'missing_expert': model['physical_to_logical_map'][0] = [0] * 6
    if change == 'invalid_expert': model['physical_to_logical_map'][0][0] = 4
    if change == 'bool_expert': model['physical_to_logical_map'][0][0] = True
    if change == 'row_width': model['physical_to_logical_map'][0].pop()
    if change == 'record_kind': model['record_kind'] = 'proposal'
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read(path)


def test_equal_size_file_replacement_is_not_cached(tmp_path):
    path = tmp_path / 'map.json'
    doc = payload()
    path.write_text(json.dumps(doc))
    old = read(path)
    row = doc['model_maps']['HYV4ForCausalLM']['physical_to_logical_map'][0]
    row[0], row[1] = row[1], row[0]
    path.write_text(json.dumps(doc))
    new = read(path)
    assert old.source_sha256 != new.source_sha256
    assert old.layer_map(0) != new.layer_map(0)


def test_other_models_do_not_gain_legacy_parser_and_v41_stays_strict(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(payload()))
    with pytest.raises(ValueError, match='HY4'):
        load_hy4_plan(path, model_key='DeepseekV41ForCausalLM',
            expected_model_name='test', expected_shape=(2, 6),
            num_logical_experts=4, num_redundant_experts=2)
    with pytest.raises(ValueError, match='format'):
        load_static_eplb_plan(path, model_key='HYV4ForCausalLM',
            expected_model_name='test', expected_shape=(2, 6),
            num_logical_experts=4, num_redundant_experts=2)
