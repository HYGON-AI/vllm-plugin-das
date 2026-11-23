"""CPU-only admission and installed-Core/source-override provenance."""
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import tomllib


REQUIRED_EXTENSIONS = {'das_hcu', 'das_hcu_dsv41', 'das_hcu_v41_indexer',
                       'das_hcu_v41_pcp', 'das_hcu_fp4_packer'}


def check_entries(group, declared, actual, extra_required=()):
    """Check this model's required discovery; report unrelated missing families."""
    required = REQUIRED_EXTENSIONS | set(extra_required) if group == 'hcu_adapter.extensions' else set(declared)
    differences = []
    for name, value in declared.items():
        matches = [v for n, v in actual if n == name]
        if matches == [value]:
            continue
        if group.startswith('vllm.') or (group == 'hcu_adapter.extensions' and (name in required or matches)):
            raise RuntimeError('Installed vLLM entry point metadata differs: '+group+'/'+name)
        differences.append(group+'/'+name)
    return differences


def check_model(raw, model_format):
    text = raw.get('language_config', raw.get('text_config', raw))
    quant = text.get('quantization_config', raw.get('quantization_config', {}))
    if model_format == 'channel-fp8':
        if quant.get('quant_method') != 'fp8' or quant.get('is_per_channel') is not True:
            raise RuntimeError('Expected the channel-FP8 checkpoint')
    elif model_format == 'ct-w4a8':
        groups = quant.get('config_groups', {})
        if quant.get('quant_method') != 'compressed-tensors':
            raise RuntimeError('Expected compressed-tensors W4A8 checkpoint')
        for name in ('group_int4_routed_experts', 'group_int4_mtp_experts'):
            group = groups.get(name, {})
            w, a = group.get('weights', {}), group.get('input_activations') or {}
            if not (w.get('num_bits') == 4 and w.get('type') == 'int'
                    and w.get('strategy') == 'channel' and w.get('symmetric') is True
                    and w.get('dynamic') is False and w.get('group_size') in (None, -1)
                    and a.get('num_bits') == 8 and a.get('type') == 'int'
                    and a.get('strategy') == 'token' and a.get('dynamic') is True
                    and a.get('symmetric') is True
                    and group.get('format', quant.get('format')) == 'pack-quantized'):
                raise RuntimeError('Unsupported W4A8 expert schema: '+name)
    else:
        raise RuntimeError('Unknown checkpoint format: '+model_format)
    return text


def main():
    project, profile_file = map(Path, sys.argv[1:3])
    profile = json.loads(profile_file.read_text())
    additional_config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    model_format = profile.get('model_format', 'channel-fp8')
    extra_required = {'das_hcu_ct_w4a8'} if model_format == 'ct-w4a8' else set()
    installed_extensions = {e.name for e in md.entry_points(group='hcu_adapter.extensions')}
    if not extra_required <= installed_extensions:
        raise RuntimeError('Missing W4A8 entry points: '+str(extra_required-installed_extensions))
    origins = {}
    for module, expected in {
        'hcu_adapter': project / 'adapter-das/hcu_adapter/__init__.py',
        'hcu_vllm_das': project / 'vllm-plugin-das/adapter_extension/src/hcu_vllm_das/__init__.py',
        'vllm': Path(md.distribution('vllm').locate_file('vllm/__init__.py')),
    }.items():
        spec = importlib.util.find_spec(module)
        if not spec or Path(spec.origin).resolve() != expected.resolve():
            raise RuntimeError('Wrong runtime module origin: ' + module)
        origins[module] = str(expected.resolve())
    unrelated_entrypoint_differences = []
    for rel in ('adapter-das/packaging/hcu-adapter/pyproject.toml',
                'vllm-plugin-das/adapter_extension/pyproject.toml'):
        declared = tomllib.loads((project / rel).read_text())['project']
        for group, entries in declared.get('entry-points', {}).items():
            actual = [(e.name, e.value) for e in md.entry_points(group=group)]
            unrelated_entrypoint_differences.extend(check_entries(group, entries, actual, extra_required))
    model = Path(profile['model'])
    raw = json.loads((model / 'config.json').read_text())
    text = check_model(raw, model_format)
    weight_index = model / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(weight_index.read_text())['weight_map'].values()))
    for shard in shards:
        if not (model / shard).is_file():
            raise FileNotFoundError(model / shard)
    from vllm.config.engram import EngramConfig
    from hcu_vllm_das.v41_pcp import bind_adapter_classes, validate_config
    from hcu_vllm_das.adapter_api import adapter_module

    validate_fp4 = adapter_module("fp4_indexer").validate_config
    cfg = NS(model_config=NS(hf_config=NS(**text), hf_text_config=NS(**text), enforce_eager=True),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=8, decode_context_parallel_size=1, data_parallel_size=1,
            enable_dbo=False, enable_eplb=False, enable_expert_parallel=True,
            enable_elastic_ep=False, expert_placement_strategy='linear',
            all2all_backend='deepep_high_throughput'),
        cache_config=NS(block_size=64, swa_bounded_replay=True),
        attention_config=NS(indexer_sparse_logits=False, indexer_kv_dtype='mxfp4', backend=None,
                            resolve_indexer_kv_dtype=lambda default: 'mxfp4'),
        kernel_config=NS(monokernel=False, enable_jit_warmup=False, moe_backend='deep_gemm'),
        use_v2_model_runner=True, additional_config=additional_config,
        speculative_config=None, kv_transfer_config=None, lora_config=None,
        engram_config=EngramConfig(cpu_offload=True, dp_shared_memory=False,
                                   embedding_across_dp=False, use_thp=False))
    if profile.get('dspark', False):
        cfg.scheduler_config = NS(max_num_seqs=128, max_num_batched_tokens=32768)
        cfg.speculative_config = NS(method='dspark', num_speculative_tokens=5,
            enable_adaptive_verification=False, draft_parallel_config=NS(tensor_parallel_size=1),
            target_parallel_config=cfg.parallel_config,
            draft_model_config=NS(enforce_eager=True, hf_config=NS(**text)))
    bind_adapter_classes()
    for split in ((True,) if additional_config else (False, True)):
        cfg.additional_config = dict(additional_config)
        if split:
            cfg.additional_config['hcu_layer_split'] = {'enabled': True}
        assert validate_config(cfg) and validate_fp4(cfg) is not None
    eplb_receipt = None
    if additional_config:
        from hcu_vllm_das.static_eplb import load_static_eplb_plan
        from hcu_vllm_das.static_eplb_runtime import read_options, validate_config as validate_eplb

        validate_eplb(cfg)
        options = read_options(cfg)
        logical = text['n_routed_experts']
        placement = load_static_eplb_plan(options.expert_map_path,
            model_key='DeepseekV41LLMForCausalLM', expected_model_name=profile['model'],
            expected_shape=(text['num_hidden_layers'], logical + options.num_redundant_experts),
            num_logical_experts=logical, num_redundant_experts=options.num_redundant_experts)
        eplb_receipt = dict(map=str(options.expert_map_path), fingerprint=placement.fingerprint())
    from hcu_vllm_das.fp4_indexer import check_native_contract
    check_native_contract()
    if model_format == 'ct-w4a8':
        from hcu_vllm_das.moe.ct_w4a8_runtime import get_ops, validate_ops
        # The selected PCP8 service uses the contiguous DeepEP-HT path.
        # Check both activation variants without allocating a device tensor.
        for limit in (None, 10.0):
            validate_ops(get_ops(), 'deepep_high_throughput', limit)
    import torch
    assert not torch.cuda.is_initialized()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    unit = tokenizer.encode('This is a fixed input for an Indexer memory test.\n', add_special_tokens=False)
    unit = [t for t in unit if t not in tokenizer.all_special_ids]
    if not unit:
        raise RuntimeError('No ordinary input tokens')
    vendor = json.loads(Path(profile['vendor_manifest']).read_text())
    for record in vendor.values():
        for filename, expected in record['binary_sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Private provider binary changed: ' + filename)
    print('PREFLIGHT_JSON=' + json.dumps(dict(origins=origins, prompt_unit=unit,
        versions={p: md.version(p) for p in ('vllm', 'torch', 'triton', 'hcu-adapter', 'hcu-vllm-das-plugin')},
        model_config_sha256=hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256(weight_index.read_bytes()).hexdigest(),
        shards=[dict(name=s, bytes=(model/s).stat().st_size, mtime_ns=(model/s).stat().st_mtime_ns) for s in shards],
        checkpoint_full_hash=False, model_format=model_format, dspark=profile.get('dspark', False),
        static_eplb=eplb_receipt,
        w4a8_ht_provider_api_checked=model_format == 'ct-w4a8', vendor=vendor, cuda_initialized=False,
        unrelated_entrypoint_differences=unrelated_entrypoint_differences)))


if __name__ == '__main__':
    main()
