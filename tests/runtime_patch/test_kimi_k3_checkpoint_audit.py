# SPDX-License-Identifier: Apache-2.0
"""Negative checkpoint cases for the CPU-only metadata and scale audit."""
import importlib.util
import json
from pathlib import Path
import struct
import sys

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
    schema = audit_module.language_schema(config)
    entries = {}
    for name, value in schema.items():
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
    w8a8_modules = sorted({
        name.removesuffix('.weight_scale').removesuffix('.weight')
        for name, spec in schema.items()
        if spec['route'] in ('w8a8', 'latent_w8a8', 'mla_fusion', 'mlp_fusion',
                             'mixed_kda_w8a8')
    })
    floating_linear_modules = sorted({
        name.removesuffix('.weight_scale').removesuffix('.weight')
        for name, spec in schema.items()
        if spec['route'] == 'float_w8a8_selectable'
    })
    (root / 'config.json').write_text(json.dumps(dict(text_config=config,
        compression_config={
            'quant_method': 'slimquant_w4a8',
            'mixed_precision': {'w8a8_include': {'modules': w8a8_modules}},
            'ignore': floating_linear_modules,
        })))
    (root / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name:'model.safetensors' for name in header}}))


def test_small_full_contract_passes(tmp_path):
    _write(tmp_path)
    result = audit_module.audit(tmp_path)
    assert result['passed'] and not result['errors']
    assert result['w8a8_scale_tensors_fully_read'] > 0
    assert result['model_constructed'] is False
    assert sum(result['route_counts'].values()) == result['language_tensor_count']


def test_audit_reports_missing_shard_instead_of_raising(tmp_path):
    _write(tmp_path)
    (tmp_path / 'model.safetensors').unlink()

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any('Unable to read shard: model.safetensors' in error
               for error in result['errors'])


def test_audit_reports_short_safetensors_preamble_instead_of_raising(tmp_path):
    _write(tmp_path)
    (tmp_path / 'model.safetensors').write_bytes(b'\0' * 7)

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any(
        'Invalid safetensors preamble: model.safetensors: '
        'expected 8 bytes, got 7' in error
        for error in result['errors']
    )


def test_cli_writes_structured_report_for_missing_shard(tmp_path, monkeypatch):
    _write(tmp_path)
    (tmp_path / 'model.safetensors').unlink()
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any('Unable to read shard: model.safetensors' in error
               for error in report['errors'])


@pytest.mark.parametrize('case', [
    'missing', 'unreadable', 'truncated', 'invalid_json', 'non_object',
    'missing_text_config', 'null_text_config', 'non_object_text_config',
    'incomplete_text_config', 'malformed_nested_config',
    'invalid_expert_count', 'invalid_quantization_config',
])
def test_cli_writes_structured_report_for_invalid_config(tmp_path, monkeypatch, case):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    if case == 'missing':
        config_path.unlink()
    elif case == 'truncated':
        config_path.write_text('{"text_config":')
    elif case == 'invalid_json':
        config_path.write_text('not json')
    elif case == 'non_object':
        config_path.write_text('[]')
    elif case == 'missing_text_config':
        config_path.write_text(json.dumps({'compression_config': {}}))
    elif case == 'null_text_config':
        config_path.write_text(json.dumps({'text_config': None}))
    elif case == 'non_object_text_config':
        config_path.write_text(json.dumps({'text_config': []}))
    elif case == 'incomplete_text_config':
        config_path.write_text(json.dumps({
            'text_config': {},
            'compression_config': {'quant_method': 'slimquant_w4a8'},
        }))
    elif case == 'malformed_nested_config':
        raw = json.loads(config_path.read_text())
        raw['text_config']['linear_attn_config'] = []
        config_path.write_text(json.dumps(raw))
    elif case == 'invalid_expert_count':
        raw = json.loads(config_path.read_text())
        raw['text_config']['num_experts'] = 'two'
        config_path.write_text(json.dumps(raw))
    elif case == 'invalid_quantization_config':
        raw = json.loads(config_path.read_text())
        raw['quantization_config'] = []
        config_path.write_text(json.dumps(raw))
    elif case == 'unreadable':
        original_read_text = Path.read_text

        def deny_config_read(path, *args, **kwargs):
            if path == config_path:
                raise PermissionError('fixture unreadable config')
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', deny_config_read)

    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False
    assert report['errors']
    assert report['mapping_contracts'] == {}


@pytest.mark.parametrize('case', ['missing', 'unreadable', 'invalid_json'])
def test_cli_writes_structured_report_for_invalid_index(tmp_path, monkeypatch, case):
    _write(tmp_path)
    index_path = tmp_path / 'model.safetensors.index.json'
    if case == 'missing':
        index_path.unlink()
    elif case == 'invalid_json':
        index_path.write_text('{"weight_map":')
    elif case == 'unreadable':
        original_read_text = Path.read_text

        def deny_index_read(path, *args, **kwargs):
            if path == index_path:
                raise PermissionError('fixture unreadable index')
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', deny_index_read)

    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False
    assert any('model.safetensors.index.json' in error for error in report['errors'])


@pytest.mark.parametrize('file_name', [
    'config.json', 'model.safetensors.index.json',
])
@pytest.mark.parametrize('digits,valid', [(101, True), (5001, False)])
def test_cli_reports_json_integer_digit_limit(tmp_path, monkeypatch, file_name,
                                              digits, valid):
    _write(tmp_path)
    path = tmp_path / file_name
    text = path.read_text()
    path.write_text(text[:-1] + ',"unused":' + '7' * digits + '}')
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == (0 if valid else 1)
    report = json.loads(output.read_text())
    assert report['passed'] is valid
    if not valid:
        assert any(file_name in error for error in report['errors'])


@pytest.mark.parametrize('bad_shard', [None, 0, False, [], {}, '', '\0'])
def test_cli_reports_invalid_shard_names(tmp_path, monkeypatch, bad_shard):
    _write(tmp_path)
    index_path = tmp_path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    # Keep valid entries too: invalid values must not reach grouping or sorting.
    name = next(iter(index['weight_map']))
    index['weight_map'][name] = bad_shard
    index_path.write_text(json.dumps(index))
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any('Invalid weight_map shard name' in error
               for error in report['errors'])


@pytest.mark.parametrize('weight_map', [None, [], 0, {'': 'model.safetensors'}])
def test_audit_reports_invalid_weight_map(tmp_path, weight_map):
    _write(tmp_path)
    (tmp_path / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': weight_map})
    )

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any('Invalid weight_map' in error for error in result['errors'])


@pytest.mark.parametrize('case,message', [
    ('escaping_path', 'Shard path escapes checkpoint'),
    ('oversized_header', 'Unreasonable safetensors header size'),
])
def test_cli_reports_rejected_shard_paths_and_headers(tmp_path, monkeypatch, case, message):
    _write(tmp_path)
    if case == 'escaping_path':
        index_path = tmp_path / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        index['weight_map']['vision_tower.invalid.weight'] = '../outside.safetensors'
        index_path.write_text(json.dumps(index))
    else:
        (tmp_path / 'model.safetensors').write_bytes(struct.pack('<Q', 100_000_001))
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any(message in error for error in report['errors'])


def test_audit_reports_short_scale_read(tmp_path, monkeypatch):
    _write(tmp_path)
    original_open = Path.open

    class ShortScaleReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            # Header reads start at 0/8. Scale reads follow the header and
            # simulate a file truncated after its ranges were validated.
            position = self.stream.tell()
            data = self.stream.read(size)
            return data[:-1] if position > 8 and data else data

    def short_read(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return ShortScaleReader(stream) if path.suffix == '.safetensors' else stream

    monkeypatch.setattr(Path, 'open', short_read)

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any('Short scale tensor read' in error for error in result['errors'])


@pytest.mark.parametrize('source', ['root', 'text_config'])
@pytest.mark.parametrize('effective_config_valid', [False, True])
def test_audit_uses_runtime_quantization_config_priority(
    tmp_path, source, effective_config_valid
):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    compression = raw['compression_config']
    effective = json.loads(json.dumps(compression))
    if not effective_config_valid:
        effective['mixed_precision']['w8a8_include']['modules'] = []
    else:
        compression['mixed_precision']['w8a8_include']['modules'] = []
    if source == 'root':
        raw['quantization_config'] = effective
    else:
        raw['text_config']['quantization_config'] = effective
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert result['passed'] is effective_config_valid
    if not effective_config_valid:
        assert any(
            'W8A8 route not selected by effective quantization config' in error
            for error in result['errors']
        )


def test_audit_accepts_packed_experts_without_ignore(tmp_path):
    _write(tmp_path)

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']
    assert result['route_counts']['packed_expert'] > 0


@pytest.mark.parametrize('selector', [
    'language_model.model.layers.1.block_sparse_moe.experts',
    r're:language_model\.model\.layers\.1\.block_sparse_moe\.experts$',
])
def test_audit_rejects_ignored_packed_experts(tmp_path, selector):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    raw['compression_config']['ignore'].append(selector)
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any(
        'Packed expert route ignored by effective quantization config' in error
        and 'language_model.model.layers.1.block_sparse_moe.experts' in error
        for error in result['errors']
    )


@pytest.mark.parametrize(
    'case', ['empty_include', 'missing_kda_branches', 'ignored_kda_branch']
)
def test_audit_rejects_w8a8_routes_not_selected_by_effective_config(tmp_path, case):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    quant = raw['compression_config']
    modules = quant['mixed_precision']['w8a8_include']['modules']
    if case == 'empty_include':
        modules.clear()
    elif case == 'missing_kda_branches':
        modules.remove('language_model.model.layers.0.self_attn.k_proj')
        modules.remove('language_model.model.layers.0.self_attn.v_proj')
    else:
        quant['ignore'] = [r're:.*layers\.0\.self_attn\.k_proj(?:\.|$)']
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any('W8A8 route not selected by effective quantization config' in error
               for error in result['errors'])
    if case == 'empty_include':
        assert result['route_counts'].get('mixed_kda_w8a8', 0) == 0


def test_audit_accepts_runtime_fused_module_selector(tmp_path):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    modules = raw['compression_config']['mixed_precision']['w8a8_include']['modules']
    gate = 'language_model.model.layers.0.mlp.gate_proj'
    up = 'language_model.model.layers.0.mlp.up_proj'
    modules.remove(gate)
    modules.remove(up)
    modules.append('language_model.model.layers.0.mlp.gate_up_proj')
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']


@pytest.mark.parametrize('projection', ['f_b_proj', 'o_proj'])
def test_audit_rejects_float_projection_selected_by_w8a8(tmp_path, projection):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    quant = raw['compression_config']
    module = f'language_model.model.layers.0.self_attn.{projection}'
    quant['mixed_precision']['w8a8_include']['modules'].append(module)
    quant['ignore'].remove(module)
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any('Floating projection selected by effective W8A8 config' in error
               and module in error for error in result['errors'])


@pytest.mark.parametrize('projection', ['f_b_proj', 'o_proj'])
def test_ignore_keeps_float_projection_out_of_w8a8(tmp_path, projection):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    quant = raw['compression_config']
    module = f'language_model.model.layers.0.self_attn.{projection}'
    quant['mixed_precision']['w8a8_include']['modules'].append(module)
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']


def test_audit_accepts_runtime_regex_unicode_property_selector(tmp_path):
    _write(tmp_path)
    config_path = tmp_path / 'config.json'
    raw = json.loads(config_path.read_text())
    modules = raw['compression_config']['mixed_precision']['w8a8_include']['modules']
    modules.remove('language_model.model.layers.0.self_attn.q_proj')
    modules.append(
        r're:language_model\.model\.layers\.\p{Nd}+\.self_attn\.q_proj$'
    )
    config_path.write_text(json.dumps(raw))

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']


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


@pytest.mark.parametrize('projection', ['w1', 'w2', 'w3'])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), float('-inf')])
def test_audit_rejects_nonfinite_packed_expert_scales(
    tmp_path, projection, bad_value
):
    def mutate(entries):
        key = (
            'language_model.model.layers.1.block_sparse_moe.experts.0.'
            f'{projection}.weight_scale'
        )
        entries[key]['value'] = bad_value

    _write(tmp_path, mutate)

    result = audit_module.audit(tmp_path)

    assert not result['passed']
    assert any(
        'Invalid packed-expert scale values' in error
        and f'.{projection}.weight_scale' in error
        for error in result['errors']
    )


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


@pytest.mark.parametrize('metadata', [None, {'format': 'pt'}])
def test_cli_rejects_reserved_metadata_in_weight_map(tmp_path, monkeypatch, metadata):
    from safetensors import safe_open

    _write(tmp_path)
    valid = audit_module.audit(tmp_path)
    _rewrite_header(tmp_path, lambda header: header.update(__metadata__=metadata))
    with safe_open(tmp_path / 'model.safetensors', framework='numpy') as reader:
        assert len(reader.keys()) == valid['language_tensor_count']
    index_path = tmp_path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    index['weight_map']['__metadata__'] = 'model.safetensors'
    index_path.write_text(json.dumps(index))
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any('Invalid weight_map tensor name' in error and '__metadata__' in error
               for error in report['errors'])
    assert report['language_tensor_count'] == valid['language_tensor_count']
    assert report['w8a8_scale_tensors_fully_read'] == valid['w8a8_scale_tensors_fully_read']


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


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_cli_rejects_nonstandard_json_constants_in_unused_header_fields(
    tmp_path, monkeypatch, value
):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    name = 'language_model.model.embed_tokens.weight'
    _rewrite_header(tmp_path, lambda header: header[name].update(extra=value))
    with pytest.raises(SafetensorError):
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any('Invalid safetensors header' in error and 'JSON constant' in error
               for error in report['errors'])


@pytest.mark.parametrize('value', [0.125, None])
def test_audit_accepts_standard_json_values_in_unused_header_fields(tmp_path, value):
    from safetensors import safe_open

    _write(tmp_path)
    name = 'language_model.model.embed_tokens.weight'
    _rewrite_header(tmp_path, lambda header: header[name].update(extra=value))
    with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
        pass

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']


@pytest.mark.parametrize('depth,reader_accepts', [
    (32, True), (125, True), (126, False), (200, False),
    (2000, False), (10000, False),
])
def test_cli_matches_reader_header_nesting_limit(tmp_path, monkeypatch,
                                                 depth, reader_accepts):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = raw[8:8 + size].decode('utf-8')
    offsets = header.index('"data_offsets": [')
    offsets_end = header.index(']', offsets) + 1
    encoded = (
        header[:offsets_end] + ',"extra":' + '[' * depth + '0' + ']' * depth
        + header[offsets_end:]
    ).encode('utf-8')
    _rewrite_raw_header(tmp_path, encoded)

    if reader_accepts:
        with safe_open(path, framework='numpy'):
            pass
    else:
        with pytest.raises(SafetensorError):
            with safe_open(path, framework='numpy'):
                pass
    if depth == 200:
        # This is the important gap: Python accepts it, serde_json rejects it.
        json.loads(encoded)

    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == (0 if reader_accepts else 1)
    report = json.loads(output.read_text())
    assert report['passed'] is reader_accepts
    if not reader_accepts:
        assert any('JSON nesting exceeds safetensors limit' in error
                   for error in report['errors'])


@pytest.mark.parametrize('token,valid', [
    ('1e309', False), ('-1e309', False),
    (r'"\ud800"', False), (r'"\udc00"', False),
    ('1e-999', True), ('-0', True), ('"NaN"', True),
    (r'"\ud800\udc00"', True),
    ('123456789012345678901234567890', True),
    pytest.param('1' + '0' * 400, False, id='huge_integer'),
    pytest.param('-1' + '0' * 400, False, id='huge_negative_integer'),
    pytest.param('1' + '0' * 308, True, id='large_finite_integer'),
    ('{"nested":[NaN]}', False),
])
def test_cli_matches_reader_for_json_number_and_unicode_boundaries(
    tmp_path, monkeypatch, token, valid
):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    name = 'language_model.model.embed_tokens.weight'
    _rewrite_header(tmp_path, lambda header: header[name].update(extra='TOKEN_PLACEHOLDER'))
    raw = (tmp_path / 'model.safetensors').read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    encoded = raw[8:8 + size].replace(b'"TOKEN_PLACEHOLDER"', token.encode('ascii'))
    _rewrite_raw_header(tmp_path, encoded)
    if valid:
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    else:
        with pytest.raises(SafetensorError):
            with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
                pass
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == (0 if valid else 1)
    report = json.loads(output.read_text())
    assert report['passed'] is valid
    if not valid:
        assert any('Invalid safetensors header' in error for error in report['errors'])


@pytest.mark.parametrize('encoding', [
    'utf-16-le', 'utf-16', 'utf-32-le', 'utf-32', 'utf-8-sig',
])
def test_audit_rejects_non_utf8_safetensors_headers(tmp_path, encoding):
    from safetensors import SafetensorError, safe_open

    _write(tmp_path)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header_text = raw[8:8 + size].decode('utf-8')
    encoded_header = header_text.encode(encoding)
    _rewrite_raw_header(tmp_path, encoded_header)

    with pytest.raises(SafetensorError):
        with safe_open(path, framework='numpy'):
            pass
    result = audit_module.audit(tmp_path)
    assert not result['passed']
    assert any('Invalid safetensors header' in error for error in result['errors'])


@pytest.mark.parametrize('prefix,suffix', [('', ''), (' \n', '\t\n')])
def test_audit_accepts_utf8_safetensors_headers_with_json_whitespace(
    tmp_path, prefix, suffix
):
    from safetensors import safe_open

    _write(tmp_path)
    path = tmp_path / 'model.safetensors'
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header_text = raw[8:8 + size].decode('utf-8')
    encoded_header = (prefix + header_text + suffix).encode('utf-8')
    _rewrite_raw_header(tmp_path, encoded_header)

    with safe_open(path, framework='numpy'):
        pass
    result = audit_module.audit(tmp_path)
    assert result['passed'] and not result['errors']


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


@pytest.mark.parametrize('field,message', [
    ('shape', 'Invalid tensor shape'), ('data_offsets', 'Invalid data_offsets'),
])
@pytest.mark.parametrize('token,valid', [('0', True), ('-0', False)])
def test_cli_matches_reader_for_negative_zero_in_usize_fields(
    tmp_path, monkeypatch, field, message, token, valid
):
    from safetensors import SafetensorError, safe_open

    name = 'vision_tower.empty.weight'
    _write(tmp_path, lambda entries: entries.update({name: dict(shape=[0], dtype='BF16')}))
    values = [0, 'TOKEN_PLACEHOLDER'] if field == 'shape' else ['TOKEN_PLACEHOLDER', 0]
    _rewrite_header(tmp_path, lambda header: header[name].update({field: values}))
    raw = (tmp_path / 'model.safetensors').read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    encoded = raw[8:8 + size].replace(b'"TOKEN_PLACEHOLDER"', token.encode('ascii'))
    _rewrite_raw_header(tmp_path, encoded)
    if valid:
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    else:
        with pytest.raises(SafetensorError):
            with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
                pass
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == (0 if valid else 1)
    report = json.loads(output.read_text())
    assert report['passed'] is valid
    if not valid:
        assert any(message in error for error in report['errors'])


@pytest.mark.parametrize('shape,message', [
    ([0, 2**64], 'Invalid tensor shape'),
    ([0, 2**100], 'Invalid tensor shape'),
    ([2**100, 0], 'Invalid tensor shape'),
    ([2**63, 2, 0], 'Tensor element count overflow'),
])
def test_cli_rejects_empty_tensor_shape_overflow(tmp_path, monkeypatch, shape, message):
    from safetensors import SafetensorError, safe_open

    name = 'vision_tower.empty.weight'
    _write(tmp_path, lambda entries: entries.update({name: dict(shape=[0], dtype='BF16')}))
    _rewrite_header(tmp_path, lambda header: header[name].update(shape=shape))
    with pytest.raises(SafetensorError):
        with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
            pass
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', [
        'audit_kimi_k3_checkpoint.py', str(tmp_path), '--output', str(output)
    ])

    assert audit_module.main() == 1
    report = json.loads(output.read_text())
    assert not report['passed']
    assert any(message in error and name in error for error in report['errors'])


@pytest.mark.parametrize('shape', [
    [0, 2], [2, 0], [0, 2**64 - 1], [2**64 - 1, 0], [0, 2**63, 2],
])
def test_audit_accepts_reader_supported_empty_tensor_shapes(tmp_path, shape):
    from safetensors import safe_open

    name = 'vision_tower.empty.weight'
    _write(tmp_path, lambda entries: entries.update({name: dict(shape=[0], dtype='BF16')}))
    _rewrite_header(tmp_path, lambda header: header[name].update(shape=shape))
    with safe_open(tmp_path / 'model.safetensors', framework='numpy'):
        pass

    result = audit_module.audit(tmp_path)

    assert result['passed'] and not result['errors']


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
