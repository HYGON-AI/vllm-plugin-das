# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Read the exact HY4 v0.25.1 cookbook map without inventing provenance."""
import hashlib
import json
from pathlib import Path

from .static_eplb import StaticEplbPlan, load_static_eplb_plan

_KEYS = frozenset({'HYV4ForCausalLM', 'HYV4MTP'})


def load_hy4_plan(path, *, model_key, expected_model_name, expected_shape,
                  num_logical_experts, num_redundant_experts):
    if model_key not in _KEYS:
        raise ValueError('legacy static placement is restricted to native HY4 map keys')
    if (type(expected_shape) is not tuple or len(expected_shape) != 2
            or any(type(n) is not int or n < 1 for n in expected_shape)):
        raise ValueError('HY4 static map expected shape must contain two positive integers')
    canonical = Path(path).expanduser().resolve(strict=True)
    raw = canonical.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError('HY4 static map must be a JSON object')
    if 'format' in document:
        # Current calibration artifacts retain their original strict format,
        # proposal and checkpoint-name checks. Never silently downgrade them.
        return load_static_eplb_plan(canonical, model_key=model_key,
            expected_model_name=expected_model_name, expected_shape=expected_shape,
            num_logical_experts=num_logical_experts,
            num_redundant_experts=num_redundant_experts)
    if (set(document) != {'version', 'model_maps'}
            or type(document['version']) is not int or document['version'] != 2):
        raise ValueError('unsupported legacy HY4 static map schema/version')
    maps = document['model_maps']
    if not isinstance(maps, dict) or not set(maps).issubset(_KEYS) or model_key not in maps:
        raise ValueError('legacy HY4 static map has invalid model keys')
    selected = maps[model_key]
    expected = dict(model_class=model_key, num_moe_layers=expected_shape[0],
        num_logical_experts=num_logical_experts,
        num_physical_experts=expected_shape[1],
        num_redundant_experts=num_redundant_experts)
    if not isinstance(selected, dict) or set(selected) != set(expected) | {'physical_to_logical_map'}:
        raise ValueError('legacy HY4 static map metadata fields do not match the cookbook schema')
    for name, value in expected.items():
        if type(selected[name]) is not type(value) or selected[name] != value:
            raise ValueError('legacy HY4 static map metadata mismatch: ' + name)
    rows = selected['physical_to_logical_map']
    if (not isinstance(rows, list) or len(rows) != expected_shape[0]
            or any(not isinstance(row, list) for row in rows)):
        raise ValueError('legacy HY4 static map layer shape mismatch')
    # The old artifact carries neither model_name nor record_kind. Here
    # model_name identifies the intended runtime target, not a recorded binding;
    # conservatively mark initial, never promote it to a calibration proposal.
    return StaticEplbPlan(model_key=model_key, model_name=expected_model_name,
        source_path=str(canonical), source_sha256=hashlib.sha256(raw).hexdigest(),
        record_kind='initial', _map_values=tuple(tuple(row) for row in rows),
        num_logical_experts=num_logical_experts,
        num_physical_experts=expected_shape[1],
        num_redundant_experts=num_redundant_experts)
