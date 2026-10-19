# SPDX-License-Identifier: Apache-2.0
"""Negative checkpoint cases for the CPU-only metadata and scale audit."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest

_PATH = Path(__file__).resolve().parents[2] / 'tools/audit_kimi_k3_checkpoint.py'
_spec = importlib.util.spec_from_file_location('k3_checkpoint_audit', _PATH)
audit_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_module)


def _write(root, mutate=None):
    config = dict(hidden_size=4, vocab_size=32, num_hidden_layers=2,
                  num_attention_heads=16, q_lora_rank=4, kv_lora_rank=4,
                  qk_rope_head_dim=2, qk_nope_head_dim=2, v_head_dim=2,
                  first_k_dense_replace=1, moe_layer_freq=1,
                  intermediate_size=32, moe_intermediate_size=32,
                  num_shared_experts=2, routed_expert_hidden_size=4, num_experts=2,
                  linear_attn_config=dict(num_heads=16, head_dim=2, full_attn_layers=[2],
                                          short_conv_kernel_size=4, use_full_rank_gate=True))
    entries = {}
    for name, value in audit_module.language_schema(config).items():
        for expert in range(value['count']):
            entries[name.replace('.experts.*.', f'.experts.{expert}.')] = dict(value)
    if mutate:
        mutate(entries)
    header, chunks, offset = {}, [], 0
    for name, value in entries.items():
        count = 1
        for dim in value['shape']:
            count *= dim
        dtype = value['dtype']
        if dtype == 'F32':
            payload = struct.pack('<f', value.get('value', .125)) * count
        else:
            payload = bytes(count * (2 if dtype == 'BF16' else 1))
        header[name] = dict(shape=value['shape'], dtype=dtype,
                            data_offsets=[offset, offset+len(payload)])
        offset += len(payload)
        chunks.append(payload)
    raw = json.dumps(header).encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(chunks))
    (root / 'config.json').write_text(json.dumps(dict(text_config=config,
        compression_config={'quant_method':'slimquant_w4a8'})))
    (root / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name:'model.safetensors' for name in header}}))


def test_small_full_contract_passes(tmp_path):
    _write(tmp_path)
    result = audit_module.audit(tmp_path)
    assert result['passed'] and not result['errors']
    assert result['w8a8_scale_tensors_fully_read'] > 0
    assert result['model_constructed'] is False


@pytest.mark.parametrize('case', ['missing_scale', 'unknown_name', 'wrong_dtype', 'bad_scale'])
def test_audit_rejects_missing_unknown_or_invalid_parameters(tmp_path, case):
    def mutate(entries):
        key = 'language_model.model.layers.0.self_attn.q_proj.weight_scale'
        if case == 'missing_scale':
            del entries[key]
        elif case == 'unknown_name':
            entries['language_model.model.unknown.weight'] = dict(shape=[1],dtype='BF16')
        elif case == 'wrong_dtype':
            entries[key]['dtype'] = 'BF16'
        else:
            entries[key]['value'] = float('nan')
    _write(tmp_path, mutate)
    result = audit_module.audit(tmp_path)
    assert not result['passed'] and result['errors']
