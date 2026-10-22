# SPDX-License-Identifier: Apache-2.0
"""Read-only K3 mixed-checkpoint name/shape/scale audit. No torch/vLLM imports.

This is an independent checkpoint contract check, not model construction or a
claim that GPU loaders/kernels have run. Vision tensors receive container
metadata validation and inventory, without architecture semantics. Supported
contract: full-rank KDA, SlimQuant per-channel
INT4 experts and W8A8 linears, as used by the Channel checkpoint.
"""
from __future__ import annotations

import argparse
import array
from collections import Counter
import fnmatch
import json
import math
import os
from pathlib import Path
import struct
import sys

import regex as re


# Safetensors storage widths, independent of torch/vLLM and model semantics.
# Packed expert INT4 values in this checkpoint are serialized as I8 tensors.
_DTYPE_BITS = {
    'BOOL': 8, 'U8': 8, 'I8': 8,
    'I16': 16, 'U16': 16, 'F16': 16, 'BF16': 16,
    'I32': 32, 'U32': 32, 'F32': 32,
    'I64': 64, 'U64': 64, 'F64': 64, 'C64': 64,
    'F8_E4M3': 8, 'F8_E5M2': 8, 'F8_E8M0': 8,
    'F8_E4M3FNUZ': 8, 'F8_E5M2FNUZ': 8,
    'F4': 4, 'F6_E2M3': 6, 'F6_E3M2': 6,
}

_CONFIG_SELECTED_ROUTES = {
    'w8a8', 'latent_w8a8', 'mla_fusion', 'mlp_fusion', 'mixed_kda_w8a8',
}

# Safetensors deserializes dimensions as native usize and checks each product.
_MAX_USIZE = (1 << (struct.calcsize('P') * 8)) - 1
# safetensors 0.8.0's serde_json reader permits at most 127 nested JSON
# containers in the complete header (including the root object).
_MAX_HEADER_JSON_NESTING = 127


def validate_tensor_extent(name, info, errors):
    """Check container metadata even for inventory-only vision tensors."""
    dtype, shape = info.get('dtype'), info.get('shape')
    if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
        errors.append(f'Invalid tensor dtype: {name}: {dtype!r}')
        return False
    if (not isinstance(shape, list)
            or any(type(dim) is not int or not 0 <= dim <= _MAX_USIZE for dim in shape)):
        errors.append(f'Invalid tensor shape: {name}: {shape!r}')
        return False
    elements = 1
    for dim in shape:
        elements *= dim
        if elements > _MAX_USIZE:
            errors.append(f'Tensor element count overflow: {name}: {shape!r}')
            return False
    bits = elements * _DTYPE_BITS[dtype]
    start, end = info['data_offsets']
    if bits % 8 or end - start != bits // 8:
        errors.append(f'Byte extent mismatch: {name}')
        return False
    return True


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
        add(attn + '.o_proj.weight', (h, heads * c['v_head_dim']),
            route='float_w8a8_selectable')
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
            add(attn + '.f_b_proj.weight', (p, hd),
                route='float_w8a8_selectable')
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
            add(attn + '.kv_b_proj.weight',
                (heads * (c['qk_nope_head_dim'] + c['v_head_dim']), kv),
                route='float_w8a8_selectable')
            add(attn + '.g_proj.weight', (heads * c['v_head_dim'], h),
                route='float_w8a8_selectable')
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


def validate_shard_ranges(header, data_size, shard, errors):
    """Require every payload byte to belong to exactly one tensor.

    Safetensors headers need not be ordered. Sorting by both endpoints also
    places empty tensors before nonempty tensors starting at the same offset.
    Include vision and unindexed entries: they share the same payload space.
    """
    ranges = []
    invalid = set()
    for name, info in header.items():
        if name == '__metadata__':
            continue
        offsets = info.get('data_offsets') if isinstance(info, dict) else None
        if (type(offsets) is not list or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            errors.append(f'Invalid data_offsets: {name}: {offsets!r}')
            invalid.add(name)
            continue
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            errors.append(f'Data offsets outside shard: {name}')
            invalid.add(name)
            continue
        if not validate_tensor_extent(name, info, errors):
            invalid.add(name)
        ranges.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start < cursor:
            errors.append(f'Overlapping tensor data: {shard}: {name}')
        elif start > cursor:
            errors.append(f'Gap in tensor data: {shard}: {cursor}..{start}')
        cursor = max(cursor, end)
    if cursor < data_size:
        errors.append(f'Trailing shard data: {shard}: {cursor}..{data_size}')
    return invalid


def _selector_matches(module, selectors):
    """Match module selectors with SlimQuant's exact/glob/``re:`` forms."""
    for selector in selectors:
        if selector.startswith('re:'):
            try:
                if re.match(selector[3:], module):
                    return True
            except re.error:
                # Invalid patterns are reported by selector validation below.
                continue
        elif fnmatch.fnmatchcase(module, selector):
            return True
    return False


def _validate_selector_patterns(label, selectors, errors):
    for selector in selectors:
        if selector.startswith('re:'):
            try:
                re.compile(selector[3:])
            except re.error as exc:
                errors.append(f'Invalid {label} selector {selector!r}: {exc}')


def validate_quantization_routes(schema, quant, errors):
    """Check that SlimQuant config selects every checkpoint W8A8 route.

    The ordinary linear config quantizes all linears when ``w8a8_include`` is
    absent, except modules matched by ``ignore``. The mixed KDA constructor is
    narrower: it only creates INT8 Q/K/V branches when an include list is
    present and selects each branch, with ``ignore`` taking precedence.
    """
    mixed_precision = quant.get('mixed_precision')
    if mixed_precision is None:
        mixed_precision = {}
    if type(mixed_precision) is not dict:
        errors.append('Invalid mixed_precision config: expected an object')
        mixed_precision = {}

    include = None
    if 'w8a8_include' in mixed_precision:
        include_config = mixed_precision.get('w8a8_include') or {}
        if type(include_config) is not dict:
            errors.append(
                'Invalid mixed_precision.w8a8_include config: expected an object'
            )
            include_config = {}
        include = include_config.get('modules') or []
        if type(include) is not list or any(type(item) is not str for item in include):
            errors.append(
                'Invalid mixed_precision.w8a8_include.modules: '
                'expected a list of strings'
            )
            include = []

    ignore = quant.get('ignore') or []
    if type(ignore) is not list or any(type(item) is not str for item in ignore):
        errors.append('Invalid compression_config.ignore: expected a list of strings')
        ignore = []
    _validate_selector_patterns(
        'mixed_precision.w8a8_include.modules', include or [], errors
    )
    _validate_selector_patterns('compression_config.ignore', ignore, errors)

    # RoutedExperts uses its ``.experts`` module prefix for quant-method
    # selection. Its packed INT4 checkpoint tensors are not governed by the
    # W8A8 include list, but an ignore match switches the runtime to an
    # unquantized method that cannot consume these packed weights and scales.
    packed_expert_modules = {
        name.split('.experts.*.', 1)[0] + '.experts'
        for name, spec in schema.items()
        if spec['route'] == 'packed_expert' and '.experts.*.' in name
    }
    for module in sorted(packed_expert_modules):
        if _selector_matches(module, ignore):
            errors.append(
                f'Packed expert route ignored by effective quantization config: '
                f'{module}'
            )

    modules = {}
    for name, spec in schema.items():
        route = spec['route']
        if route not in _CONFIG_SELECTED_ROUTES | {'float_w8a8_selectable'}:
            continue
        source = name.removesuffix('.weight_scale').removesuffix('.weight')
        destination = spec.get('destination', name)
        runtime = destination.removesuffix('.weight_scale').removesuffix('.weight')
        # Mixed KDA checks each split branch's original module prefix. Other
        # routes may be constructed as a fused module (for example gate_up_proj).
        if route == 'mixed_kda_w8a8':
            runtime = source
        modules[source, route] = runtime
    selected_routes = set()
    for (module, route), runtime in sorted(modules.items()):
        ignored = (_selector_matches(module, ignore)
                   or runtime != module and _selector_matches(runtime, ignore))
        include_match = (
            _selector_matches(module, include or [])
            or runtime != module and _selector_matches(runtime, include or [])
        )
        included = include is None or include_match
        selected = not ignored and included
        if route == 'mixed_kda_w8a8':
            selected = not ignored and include is not None and include_match
        if route == 'float_w8a8_selectable':
            if selected:
                errors.append(
                    f'Floating projection selected by effective W8A8 config: '
                    f'{module}'
                )
            continue
        if selected:
            selected_routes.add((module, route))
        else:
            errors.append(
                f'W8A8 route not selected by effective quantization config: '
                f'{module} ({route})'
            )
    return selected_routes


def _unique_object(pairs):
    """Build a JSON object without discarding duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON header key: {key}')
        result[key] = value
    return result


def _reject_json_constant(value):
    """Python accepts these tokens, but safetensors requires standard JSON."""
    raise ValueError(f'Invalid JSON constant: {value}')


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'JSON number out of range: {value}')
    return result


def _json_integer(value):
    # serde_json stores integral values as i64/u64, then falls back to a
    # finite f64. Match that behavior so oversized unknown fields cannot make
    # this parser more permissive than the safetensors reader.
    if value == '-0':
        # serde_json preserves -0 as a float, so usize fields must reject it.
        return -0.0
    result = int(value)
    if -(1 << 63) <= result <= _MAX_USIZE:
        return result
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'JSON number out of range: {value}')
    return result


def _validate_json_strings(value):
    """Reject lone surrogate escapes even in fields not used by the audit."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            item.encode('utf-8')
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _validate_json_nesting(text, max_depth=_MAX_HEADER_JSON_NESTING):
    """Reject headers deeper than the serde_json reader accepts.

    This lightweight scan runs before json.loads so very deep input cannot
    exploit a more permissive parser or trigger Python's recursion limit.
    Brackets inside JSON strings are ignored; syntax pairing remains the
    decoder's responsibility.
    """
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in '[{':
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    f'JSON nesting exceeds safetensors limit ({max_depth})'
                )
        elif char in ']}':
            depth -= 1


def _valid_metadata(header, shard, errors):
    """Match safetensors metadata: absent/null or a string-to-string object."""
    if '__metadata__' not in header:
        return
    metadata = header['__metadata__']
    if metadata is None:
        return
    if (type(metadata) is not dict
            or any(type(key) is not str or type(value) is not str
                   for key, value in metadata.items())):
        errors.append(f'Invalid safetensors metadata: {shard}')


def _effective_quantization_config(raw, text_config):
    """Resolve the checkpoint config with vLLM's native loader priority."""
    for config in (
        raw.get('quantization_config'),
        text_config.get('quantization_config'),
        raw.get('compression_config'),
    ):
        if config is not None:
            return config
    return {}


def _failed_audit_result(root, errors):
    """Return the normal report shape when config prevents schema auditing."""
    return dict(checkpoint=str(root), tensor_count=0, language_tensor_count=0,
                vision_inventory_only=0, route_counts={},
                w8a8_scale_tensors_fully_read=0, w8a8_scale_values_checked=0,
                errors=errors, passed=False, model_constructed=False,
                GPU_initialized=False, mapping_contracts={},
                limits='Schema and source routing audit; not complete live-model parameter coverage or GPU loading')


def audit(root, tp=16):
    root = Path(root)
    errors = []
    config_path = root / 'config.json'
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        return _failed_audit_result(
            root, [f'Unable to read/parse config.json: {exc}']
        )
    if type(raw) is not dict:
        return _failed_audit_result(
            root, ['Invalid config.json: expected a JSON object']
        )
    c = raw.get('text_config')
    if type(c) is not dict:
        return _failed_audit_result(
            root, ['Invalid config.json text_config: expected an object']
        )
    try:
        quant = _effective_quantization_config(raw, c)
        if type(quant) is not dict:
            raise ValueError('quantization config must be an object')
        if quant.get('quant_method') != 'slimquant_w4a8':
            raise ValueError('Expected slimquant_w4a8 checkpoint')
        schema = language_schema(c)
        expected_expert_ids = set(range(c['num_experts']))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError,
            OverflowError, ZeroDivisionError) as exc:
        return _failed_audit_result(
            root, [f'Invalid checkpoint config: {exc}']
        )
    index_path = root / 'model.safetensors.index.json'
    try:
        index_document = json.loads(index_path.read_text())
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        return _failed_audit_result(
            root, [f'Unable to read/parse model.safetensors.index.json: {exc}']
        )
    index = index_document.get('weight_map') if isinstance(index_document, dict) else None
    if not isinstance(index, dict):
        errors.append('Invalid weight_map: expected an object')
        index = {}
    shards = {}
    for name, shard in index.items():
        if not isinstance(name, str) or not name or name == '__metadata__':
            errors.append(f'Invalid weight_map tensor name: {name!r}')
            continue
        if not isinstance(shard, str) or not shard or '\0' in shard:
            errors.append(f'Invalid weight_map shard name for {name!r}: {shard!r}')
            continue
        shards.setdefault(shard, set()).add(name)
    counts = Counter()
    expert_ids = {name: set() for name in schema if '.experts.*.' in name}
    if not expert_ids:
        expected_expert_ids = set()
    routes = Counter()
    w8_scales = 0
    w8_values = 0
    vision = 0
    selected_routes = validate_quantization_routes(schema, quant, errors)
    for shard, indexed in sorted(shards.items()):
        try:
            path = (root / shard).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f'Invalid shard path: {shard!r}: {exc}')
            continue
        if not path.is_relative_to(root.resolve()):
            errors.append(f'Shard path escapes checkpoint: {shard}')
            continue
        try:
            stream = path.open('rb')
        except OSError as exc:
            errors.append(f'Unable to read shard: {shard}: {exc}')
            continue
        with stream:
            try:
                preamble = stream.read(8)
            except OSError as exc:
                errors.append(f'Unable to read shard preamble: {shard}: {exc}')
                continue
            if len(preamble) != 8:
                errors.append(
                    f'Invalid safetensors preamble: {shard}: '
                    f'expected 8 bytes, got {len(preamble)}'
                )
                continue
            size = struct.unpack('<Q', preamble)[0]
            if size > 100_000_000:
                errors.append(f'Unreasonable safetensors header size: {shard}: {size}')
                continue
            try:
                header_text = stream.read(size).decode('utf-8')
                _validate_json_nesting(header_text)
                header = json.loads(
                    header_text, object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                    parse_float=_finite_json_float,
                    parse_int=_json_integer,
                )
                _validate_json_strings(header)
            except (
                UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError
            ) as exc:
                errors.append(f'Invalid safetensors header: {shard}: {exc}')
                continue
            if type(header) is not dict:
                errors.append(f'Invalid safetensors header object: {shard}')
                continue
            _valid_metadata(header, shard, errors)
            try:
                data_size = os.fstat(stream.fileno()).st_size - 8 - size
            except OSError as exc:
                errors.append(f'Unable to stat shard: {shard}: {exc}')
                continue
            invalid_ranges = validate_shard_ranges(header, data_size, shard, errors)
            if set(header) - {'__metadata__'} != indexed:
                errors.append(f'Index/header mismatch: {shard}')
            for name in indexed:
                if name not in header or name in invalid_ranges:
                    continue
                info = header[name]
                start, end = info['data_offsets']
                if name.startswith(('vision_tower.', 'mm_projector.')):
                    vision += 1  # Inventory only; not proof of visual-model loading.
                    continue
                pattern = name
                expert = re.search(r'\.experts\.(\d+)\.', name)
                if expert:
                    expert_id = int(expert[1])
                    if expert[1] != str(expert_id) or expert_id not in expected_expert_ids:
                        errors.append(f'Invalid expert ID: {name}')
                        continue
                    pattern = name[:expert.start(1)] + '*' + name[expert.end(1):]
                spec = schema.get(pattern)
                if spec is None:
                    errors.append(f'Unknown language tensor: {name}')
                    continue
                counts[pattern] += 1
                if expert:
                    expert_ids[pattern].add(expert_id)
                module = name.removesuffix('.weight_scale').removesuffix('.weight')
                if (spec['route'] not in _CONFIG_SELECTED_ROUTES
                        or (module, spec['route']) in selected_routes):
                    routes[spec['route']] += 1
                if info['dtype'] != spec['dtype'] or info['shape'] != spec['shape']:
                    errors.append(f'Shape/dtype mismatch: {name}')
                    continue
                if name.endswith('.weight_scale'):
                    try:
                        stream.seek(8 + size + start)
                        data = stream.read(end - start)
                    except OSError as exc:
                        errors.append(
                            f'Unable to read scale tensor: {name}: {exc}'
                        )
                        continue
                    if len(data) != end - start:
                        errors.append(
                            f'Short scale tensor read: {name}: '
                            f'expected {end - start} bytes, got {len(data)}'
                        )
                        continue
                    values = array.array('f')
                    values.frombytes(data)
                    if sys.byteorder != 'little':
                        values.byteswap()
                    valid_values = all(math.isfinite(v) for v in values)
                    if spec['route'] == 'packed_expert':
                        if not valid_values:
                            errors.append(f'Invalid packed-expert scale values: {name}')
                    else:
                        if not valid_values or not all(v > 0 for v in values):
                            errors.append(f'Invalid W8A8 scale values: {name}')
                        w8_scales += 1
                        w8_values += len(values)
    for name, spec in schema.items():
        if counts[name] != spec['count']:
            errors.append(f'Missing/excess tensor: {name}: {counts[name]} != {spec["count"]}')
        if name in expert_ids and expert_ids[name] != expected_expert_ids:
            missing = sorted(expected_expert_ids - expert_ids[name])
            extra = sorted(expert_ids[name] - expected_expert_ids)
            errors.append(f'Expert ID coverage mismatch: {name}: missing={missing}, extra={extra}')
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
