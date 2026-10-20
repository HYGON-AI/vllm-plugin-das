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


def _rewrite_header(root, mutate):
    path = root / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    mutate(header)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + raw[8 + size:])


def _rewrite_raw_header(root, encoded_header):
    path = root / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    path.write_bytes(struct.pack('<Q', len(encoded_header)) + encoded_header + raw[8 + size:])


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


@pytest.mark.parametrize('case', ['past_eof', 'negative', 'reversed', 'truncated', 'vision'])
def test_audit_rejects_invalid_data_ranges(tmp_path, case):
    def add_vision(entries):
        if case == 'vision':
            entries['vision_tower.test.weight'] = dict(shape=[1], dtype='BF16', count=1)

    _write(tmp_path, add_vision)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    name = ('vision_tower.test.weight' if case == 'vision'
            else 'language_model.model.embed_tokens.weight')
    start, end = header[name]['data_offsets']
    length = end - start
    if case == 'truncated':
        payload = payload[:-1]
    else:
        header[name]['data_offsets'] = (
            [-length, 0] if case == 'negative' else
            [end, start] if case == 'reversed' else
            [10**12, 10**12 + length])
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any('Data offsets outside shard' in error for error in result['errors'])


@pytest.mark.parametrize('tensor_kind', ['language', 'vision'])
@pytest.mark.parametrize('offsets', [[0.0, 2.0], [False, 2]])
def test_audit_rejects_non_integer_data_offsets(tmp_path, tensor_kind, offsets):
    from safetensors import SafetensorError, safe_open

    name = 'language_model.model.embed_tokens.weight'
    if tensor_kind == 'vision':
        name = 'vision_tower.test.weight'
        _write(tmp_path, lambda entries: entries.update(
            {name: dict(shape=[1], dtype='BF16')}))
    else:
        _write(tmp_path)
    _rewrite_header(tmp_path, lambda header: header[name].update(data_offsets=offsets))

    with pytest.raises(SafetensorError):
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any(f'Invalid data_offsets: {name}' in error for error in result['errors'])


@pytest.mark.parametrize('metadata', [
    {'format': 123}, [], 'metadata', False,
])
def test_audit_rejects_invalid_safetensors_metadata(tmp_path, metadata):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    _rewrite_header(tmp_path, lambda header: header.update(__metadata__=metadata))
    with pytest.raises(SafetensorError):
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any('Invalid safetensors metadata' in error for error in result['errors'])


@pytest.mark.parametrize('metadata', [None, {'format': 'pt'}])
def test_audit_accepts_supported_safetensors_metadata(tmp_path, metadata):
    from safetensors import safe_open

    _write(tmp_path)
    _rewrite_header(tmp_path, lambda header: header.update(__metadata__=metadata))
    with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
        pass
    result = audit_module.audit(tmp_path)
    assert result['passed'] and not result['errors']


def test_audit_rejects_duplicate_header_keys_without_losing_first_value(tmp_path):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    name = 'language_model.model.embed_tokens.weight'
    valid = header[name]
    invalid = dict(valid, shape=[-1])
    entries = [(name, invalid)] + [
        (key, value) for key, value in header.items() if key != name
    ] + [(name, valid)]
    encoded = ('{' + ','.join(json.dumps(key) + ':' + json.dumps(value)
                              for key, value in entries) + '}').encode()
    _rewrite_raw_header(tmp_path, encoded)

    with pytest.raises(SafetensorError):
        with safe_open(path, framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any('Duplicate JSON header key' in error for error in result['errors'])


@pytest.mark.parametrize('case,message', [
    ('identical', 'Overlapping'), ('partial', 'Overlapping'),
    ('vision', 'Overlapping'), ('leading_gap', 'Gap'),
    ('internal_gap', 'Gap'), ('trailing', 'Trailing'),
])
def test_audit_rejects_non_partitioning_shard_ranges(tmp_path, case, message):
    from safetensors import SafetensorError, safe_open

    def add_vision(entries):
        if case == 'vision':
            entries['vision_tower.test.weight'] = dict(shape=[1], dtype='BF16')

    _write(tmp_path, add_vision)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    parent = 'language_model.model.layers.0.self_attn.'
    if case in ('identical', 'partial'):
        start, end = header[parent + 'q_proj.weight']['data_offsets']
        shift = int(case == 'partial')
        header[parent + 'k_proj.weight']['data_offsets'] = [start + shift, end + shift]
    elif case == 'vision':
        header['vision_tower.test.weight']['data_offsets'] = [0, 2]
    elif case in ('leading_gap', 'internal_gap'):
        split = 0 if case == 'leading_gap' else header[parent + 'k_proj.weight']['data_offsets'][0]
        payload = payload[:split] + b'\0' + payload[split:]
        for info in header.values():
            start, end = info['data_offsets']
            if start >= split:
                info['data_offsets'] = [start + 1, end + 1]
    else:
        payload += b'\0'
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    # An independent loader confirms these files are structurally invalid.
    with pytest.raises(SafetensorError):
        with safe_open(path, framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any(message in error for error in result['errors'])


def test_audit_accepts_unordered_header_with_empty_tensor(tmp_path):
    def add_empty(entries):
        entries['vision_tower.empty.weight'] = dict(shape=[0], dtype='BF16')

    _write(tmp_path, add_empty)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    encoded = json.dumps(dict(reversed(list(header.items())))).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + raw[8 + size:])
    result = audit_module.audit(tmp_path)
    assert result['passed'] and not result['errors']


@pytest.mark.parametrize('prefix', ['vision_tower', 'mm_projector'])
@pytest.mark.parametrize('case,message', [
    ('extent', 'Byte extent mismatch'), ('dtype', 'Invalid tensor dtype'),
    ('negative_shape', 'Invalid tensor shape'), ('fractional_shape', 'Invalid tensor shape'),
])
def test_vision_inventory_requires_valid_tensor_metadata(tmp_path, prefix, case, message):
    from safetensors import SafetensorError, safe_open

    name = prefix + '.test.weight'
    _write(tmp_path, lambda entries: entries.update({name: dict(shape=[1], dtype='BF16')}))

    def mutate(header):
        info = header[name]
        if case == 'extent':
            info['shape'] = [2]
        elif case == 'dtype':
            info['dtype'] = 'INVALID'
        else:
            info['shape'] = [-1] if case == 'negative_shape' else [1.0]

    _rewrite_header(tmp_path, mutate)
    # Offsets remain a valid partition: only the tensor metadata is invalid.
    with pytest.raises(SafetensorError):
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any(message in error and name in error for error in result['errors'])


@pytest.mark.parametrize('dtype,shape,byte_count', [
    ('F16', [1], 2), ('F32', [], 4), ('F64', [], 8),
    ('U8', [2], 2), ('BOOL', [2], 2), ('I16', [0], 0),
])
def test_vision_inventory_accepts_generic_valid_metadata(tmp_path, dtype, shape, byte_count):
    from safetensors import safe_open

    name = 'vision_tower.test.weight'
    _write(tmp_path, lambda entries: entries.update({name: dict(shape=[byte_count], dtype='I8')}))

    def mutate(header):
        header[name].update(dtype=dtype, shape=shape)

    _rewrite_header(tmp_path, mutate)
    with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
        pass
    result = audit_module.audit(tmp_path)
    assert result['passed'] and result['vision_inventory_only'] == 1


@pytest.mark.parametrize('projection', ['w1', 'w2', 'w3'])
@pytest.mark.parametrize('alias', ['00', '01'])
def test_audit_rejects_noncanonical_expert_ids(tmp_path, projection, alias):
    from safetensors import safe_open

    owner = 'language_model.model.layers.1.block_sparse_moe.experts.'

    def mutate(entries):
        for suffix in ('weight', 'weight_scale'):
            source = f'{owner}1.{projection}.{suffix}'
            entries[f'{owner}{alias}.{projection}.{suffix}'] = entries.pop(source)

    _write(tmp_path, mutate)
    # This is a valid safetensors container, but not valid expert coverage.
    with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
        pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any('Invalid expert ID' in error and f'.{alias}.' in error
               for error in result['errors'])
    assert any('Expert ID coverage mismatch' in error and f'.{projection}.' in error
               for error in result['errors'])
