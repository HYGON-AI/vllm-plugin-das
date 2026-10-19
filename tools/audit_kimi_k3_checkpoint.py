# SPDX-License-Identifier: Apache-2.0
"""Read-only K3 mixed-checkpoint name/shape/scale audit. No torch/vLLM imports.

This is an independent checkpoint contract check, not model construction or a
claim that GPU loaders/kernels have run. Vision is inventoried separately for
--language-model-only. Supported contract: full-rank KDA, SlimQuant per-channel
INT4 experts and W8A8 linears, as used by the Channel checkpoint.
"""
from __future__ import annotations

import argparse
import array
from collections import Counter
import json
import math
from pathlib import Path
import re
import struct
import sys


def language_schema(c):
    schema = {}
    h, vocab = c['hidden_size'], c['vocab_size']
    kda = c['linear_attn_config']
    if not kda.get('use_full_rank_gate'):
        raise ValueError('Audit requires full-rank K3 KDA')
    kh, hd = kda['num_heads'], kda['head_dim']
    p = kh * hd
    heads = c['num_attention_heads']
    kv, qr = c['kv_lora_rank'], c['q_lora_rank']

    def add(name, shape, dtype='BF16', destination=None, shard=None, count=1, route='direct'):
        schema[name] = dict(shape=list(shape), dtype=dtype, destination=destination or name,
                            shard=shard, count=count, route=route)

    def linear(name, rows, cols, destination=None, shard=None, route='w8a8'):
        add(name + '.weight', (rows, cols), 'I8',
            (destination or name) + '.weight', shard, route=route)
        add(name + '.weight_scale', (rows, 1), 'F32',
            (destination or name) + '.weight_scale', shard, route=route)
        # Registered parameter paths differ from the original checkpoint prefix
        # deliberately passed to the three mixed KDA linear constructors.
        for suffix in ('.weight', '.weight_scale'):
            schema[name + suffix]['quantization_prefix'] = (
                name if route == 'mixed_kda_w8a8' else destination or name)

    add('language_model.lm_head.weight', (vocab, h))
    add('language_model.model.embed_tokens.weight', (vocab, h))
    for name in ('norm', 'output_attn_res_norm'):
        add('language_model.model.' + name + '.weight', (h,))
    add('language_model.model.output_attn_res_proj.weight', (1, h))
    full = set(kda['full_attn_layers'])
    for i in range(c['num_hidden_layers']):
        base = f'language_model.model.layers.{i}'
        for name in ('input_layernorm', 'post_attention_layernorm',
                     'mlp_res_norm', 'self_attention_res_norm'):
            add(f'{base}.{name}.weight', (h,))
        for name in ('mlp_res_proj', 'self_attention_res_proj'):
            add(f'{base}.{name}.weight', (1, h))
        attn = base + '.self_attn'
        add(attn + '.o_proj.weight', (h, heads * c['v_head_dim']))
        if i + 1 not in full:
            for name in ('q_proj', 'k_proj', 'v_proj'):
                linear(attn + '.' + name, p, h,
                       attn + '.in_proj_qkvgfab.' + name, route='mixed_kda_w8a8')
            for name, rows, shard in (('g_proj', p, 0), ('f_a_proj', hd, 1), ('b_proj', kh, 2)):
                add(attn + '.' + name + '.weight', (rows, h),
                    destination=attn + '.in_proj_qkvgfab.fp_gate.weight',
                    shard=shard, route='mixed_kda_float')
            for shard, name in enumerate(('q_conv1d', 'k_conv1d', 'v_conv1d')):
                add(attn + '.' + name + '.weight', (p, 1, kda['short_conv_kernel_size']),
                    destination=attn + '.conv1d.weight', shard=shard, route='conv_fusion')
            add(attn + '.f_b_proj.weight', (p, hd))
            add(attn + '.o_norm.weight', (hd,))
            # Serialized K3 A_log uses head_dim entries; the installed loader
            # narrows to local head rows. The audit does not reinterpret values.
            add(attn + '.A_log', (hd,), 'F32', route='native_a_log_loader')
            add(attn + '.dt_bias', (p,), 'F32')
        else:
            linear(attn + '.q_a_proj', qr, h, attn + '.fused_qkv_a_proj', 0, 'mla_fusion')
            linear(attn + '.kv_a_proj_with_mqa', kv + c['qk_rope_head_dim'], h,
                   attn + '.fused_qkv_a_proj', 1, 'mla_fusion')
            linear(attn + '.q_b_proj', heads * (c['qk_nope_head_dim'] + c['qk_rope_head_dim']), qr)
            add(attn + '.q_a_layernorm.weight', (qr,))
            add(attn + '.kv_a_layernorm.weight', (kv,))
            add(attn + '.kv_b_proj.weight', (heads * (c['qk_nope_head_dim'] + c['v_head_dim']), kv))
            add(attn + '.g_proj.weight', (heads * c['v_head_dim'], h))
        moe = (i >= c['first_k_dense_replace'] and i % c['moe_layer_freq'] == 0)
        if not moe:
            mlp, mid = base + '.mlp', c['intermediate_size']
        else:
            mlp = base + '.block_sparse_moe.shared_experts'
            mid = c['moe_intermediate_size'] * c['num_shared_experts']
            owner = base + '.block_sparse_moe'
            latent = c['routed_expert_hidden_size']
            n = c['moe_intermediate_size']
            add(owner + '.gate.weight', (c['num_experts'], h))
            add(owner + '.gate.e_score_correction_bias', (c['num_experts'],))
            add(owner + '.routed_expert_norm.weight', (latent,))
            linear(owner + '.routed_expert_down_proj', latent, h, route='latent_w8a8')
            linear(owner + '.routed_expert_up_proj', h, latent, route='latent_w8a8')
            for name, rows, cols, dest, shard in (
                ('w1', n, latent // 2, 'w13', 'w1'),
                ('w3', n, latent // 2, 'w13', 'w3'),
                ('w2', latent, n // 2, 'w2', 'w2'),
            ):
                source = owner + '.experts.*.' + name
                destination = owner + '.experts.routed_experts.' + dest + '_weight'
                add(source + '.weight', (rows, cols), 'I8', destination, shard,
                    c['num_experts'], 'packed_expert')
                add(source + '.weight_scale', (rows, 1), 'F32', destination + '_scale', shard,
                    c['num_experts'], 'packed_expert')
        for shard, name in enumerate(('gate_proj', 'up_proj')):
            linear(mlp + '.' + name, mid, h, mlp + '.gate_up_proj', shard, 'mlp_fusion')
        linear(mlp + '.down_proj', h, mid)
    return schema


def audit(root, tp=16):
    root = Path(root)
    raw = json.loads((root / 'config.json').read_text())
    c = raw['text_config']
    quant = raw.get('compression_config', c.get('compression_config', {}))
    if quant.get('quant_method') != 'slimquant_w4a8':
        raise ValueError('Expected slimquant_w4a8 checkpoint')
    schema = language_schema(c)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    shards = {}
    for name, shard in index.items():
        shards.setdefault(shard, set()).add(name)
    counts = Counter()
    routes = Counter()
    w8_scales = 0
    w8_values = 0
    vision = 0
    errors = []
    for shard, indexed in sorted(shards.items()):
        path = (root / shard).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Shard path escapes checkpoint')
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError('Unreasonable safetensors header size')
            header = json.loads(stream.read(size))
            if set(header) - {'__metadata__'} != indexed:
                errors.append(f'Index/header mismatch: {shard}')
            for name in indexed:
                if name.startswith(('vision_tower.', 'mm_projector.')):
                    vision += 1  # Inventory only; not proof of visual-model loading.
                    continue
                pattern = name
                expert = re.search(r'\.experts\.(\d+)\.', name)
                if expert:
                    if not 0 <= int(expert[1]) < c['num_experts']:
                        errors.append(f'Invalid expert ID: {name}')
                    pattern = name[:expert.start(1)] + '*' + name[expert.end(1):]
                spec = schema.get(pattern)
                if spec is None:
                    errors.append(f'Unknown language tensor: {name}')
                    continue
                if name not in header:
                    continue
                info = header[name]
                counts[pattern] += 1
                routes[spec['route']] += 1
                if info['dtype'] != spec['dtype'] or info['shape'] != spec['shape']:
                    errors.append(f'Shape/dtype mismatch: {name}')
                    continue
                start, end = info['data_offsets']
                width = {'BF16': 2, 'F32': 4, 'I8': 1}[spec['dtype']]
                if end - start != math.prod(spec['shape']) * width:
                    errors.append(f'Byte extent mismatch: {name}')
                    continue
                if name.endswith('.weight_scale') and spec['route'] != 'packed_expert':
                    stream.seek(8 + size + start)
                    data = stream.read(end - start)
                    values = array.array('f')
                    values.frombytes(data)
                    if sys.byteorder != 'little':
                        values.byteswap()
                    if len(data) != end-start or not all(math.isfinite(v) and v > 0 for v in values):
                        errors.append(f'Invalid W8A8 scale values: {name}')
                    w8_scales += 1
                    w8_values += len(values)
    for name, spec in schema.items():
        if counts[name] != spec['count']:
            errors.append(f'Missing/excess tensor: {name}: {counts[name]} != {spec["count"]}')
    for value in (c['linear_attn_config']['num_heads'], c['num_attention_heads'],
                  c['moe_intermediate_size'], c['intermediate_size']):
        if value % tp:
            errors.append(f'TP{tp} divisibility failure: {value}')
    return dict(checkpoint=str(root), tensor_count=len(index),
                language_tensor_count=sum(counts.values()), vision_inventory_only=vision,
                route_counts=dict(routes), w8a8_scale_tensors_fully_read=w8_scales,
                w8a8_scale_values_checked=w8_values, errors=errors,
                passed=not errors, model_constructed=False, GPU_initialized=False,
                mapping_contracts=schema,
                limits='Schema and source routing audit; not complete live-model parameter coverage or GPU loading')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'mapping_contracts'}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
