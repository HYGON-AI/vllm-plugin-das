# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Portable metadata/fixture CLIs using tiny real safetensors bytes, no GPU."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments/kimi31_slimquant_preload'


def run(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS/name), *map(str, args)],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('name', ['plan_small_fixture.py', 'materialize_small_fixture.py'])
def test_source_is_required_and_help_needs_no_checkpoint(tmp_path, name):
    assert run(name, '--help').returncode == 0
    missing = run(name)
    assert missing.returncode == 2 and '--source-model' in missing.stderr
    invalid = run(name, '--source-model', tmp_path/'missing')
    assert invalid.returncode == 2 and 'model.safetensors.index.json' in invalid.stderr


def test_plan_and_materialize_accept_arbitrary_source_and_output(tmp_path):
    source = tmp_path/'checkpoint with spaces'
    source.mkdir()
    names = ['language_model.model.embed_tokens.weight',
        'language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight',
        'language_model.model.layers.0.block_sparse_moe.experts.2.w1.weight',
        'language_model.model.layers.4.self_attn.weight']
    header = {name: dict(dtype='I8', shape=[4], data_offsets=[i*4, (i+1)*4])
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    (source/'weights.safetensors').write_bytes(struct.pack('<Q', len(raw))+raw+bytes(range(16)))
    (source/'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {name: 'weights.safetensors' for name in names}}))
    (source/'config.json').write_text(json.dumps({'text_config': {
        'num_hidden_layers': 5, 'num_experts': 3, 'num_experts_per_token': 2}}))
    (source/'tokenizer.json').write_text('{}')
    plan = tmp_path/'plan.json'
    result = run('plan_small_fixture.py', '--source-model', source, '--output', plan)
    assert result.returncode == 0, result.stderr
    assert json.loads(plan.read_text())['selected_tensors'] == 2
    assert json.loads(plan.read_text())['selected_bytes'] == 8
    output = tmp_path/'custom output'/'fixture'
    result = run('materialize_small_fixture.py', '--source-model', source, '--output-dir', output)
    assert result.returncode == 0, result.stderr
    config = json.loads((output/'config.json').read_text())
    assert config['text_config']['num_hidden_layers'] == 4
    assert config['text_config']['num_experts'] == 2
    index = json.loads((output/'model.safetensors.index.json').read_text())['weight_map']
    assert set(index) == set(names[:2])
    manifest = json.loads((output/'fixture-manifest.json').read_text())
    assert manifest['source'] == str(source.resolve())
    assert manifest['tensors'][names[0]]['sha256'] == hashlib.sha256(bytes(range(4))).hexdigest()
    assert manifest['tensors'][names[1]]['sha256'] == hashlib.sha256(bytes(range(4, 8))).hexdigest()
    assert (output/'tokenizer.json').read_text() == '{}'
    repeated = run('materialize_small_fixture.py', '--source-model', source, '--output-dir', output)
    assert repeated.returncode != 0
