# SPDX-License-Identifier: Apache-2.0
"""Launch the validated four-host Step5 E23 profile from installed wheels."""
import argparse
import hashlib
import importlib.metadata as metadata
import ipaddress
import json
import os
from pathlib import Path
import sys

ENVIRONMENT = {'HSA_FORCE_FINE_GRAIN_PCIE': '1', 'OMP_NUM_THREADS': '1', 'GPU_MAX_HW_QUEUES': '4', 'VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM': '1', 'VLLM_HCU_USE_FUSE_MOE_GATE': '1', 'VLLM_HCU_USE_FLASH_ATTN': '1', 'VLLM_HCU_USE_FLASH_ATTN_UNIFIED': '1', 'VLLM_USE_PD_SPLIT': '1', 'VLLM_HCU_USE_CUSTOM_OPS': '1', 'VLLM_HCU_USE_LIGHTOP_EP_SCATTER': '1', 'VLLM_HCU_USE_CUSTOM_SILU_AND_MUL': '1', 'VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM': '1', 'VLLM_ROCM_USE_AITER_MOE': '1', 'VLLM_USE_FLASH_ATTN_FP8': '1', 'VLLM_HCU_USE_LIGHTOP_MOE_ALIGN': '1', 'FA_DECODE_NUM_SPLITS': '2', 'VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD': '0', 'VLLM_HCU_DP_CUSTOM_AG': '1', 'VLLM_HCU_STEP5_FP8_DENSE': '1', 'VLLM_HCU_STEP5_FP8_BF16_MOE': '1', 'VLLM_HCU_STEP5_SWA_FUSED_QK': '1', 'VLLM_HCU_STEP5_FUSED_GATE': '1', 'VLLM_HCU_TRITON_TOKEN_FP8_QUANT': '1', 'VLLM_ENGINE_READY_TIMEOUT_S': '3600', 'VLLM_HCU_PIN_MEMORY': '1', 'HIP_FORCE_DEV_KERNARG': '1', 'VLLM_SERVER_DEV_MODE': '1', 'VLLM_USE_V2_MODEL_RUNNER': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'NCCL_DEBUG': 'INFO', 'VLLM_HCU_STEP5_QKV_INDEXER': '1', 'VLLM_STEP4_DSA_FLASH_ATTN': 'prefill,decode', 'HCU_MTP_PREFILL_TAIL': '1536', 'HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS': '4096', 'HCU_DP_MOE_PIPE_MIN_TOKENS': '0', 'HCU_DBO_MIN_TOKENS': '4096', 'HCU_EP_DIRECT_SUM': '0', 'HCU_STEP5_EXPERT_GRID': '0', 'HCU_STEP5_FUSED_ROUTER_STREAM': '0', 'HCU_STEP5_BOUNDED_EP_WORKSPACE': '0', 'EP_FUSED_MOE_CHUNK': '32768', 'VLLM_STEP4_DSA_GRAPH_REGIONS': '8704', 'VLLM_HCU_EP_FUSED_MOE': '0', 'VLLM_HCU_EP_FUSED_COMBINE': '1', 'HCU_STEP5_DENSE_TUNING': '1', 'HCU_STEP5_PREFILL_ASM_G1': '0', 'HCU_STEP5_PREFILL_G2': '0', 'HCU_STEP5_SPARSE_RS': '0', 'VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH': '0', 'VLLM_DEEPEP_BUFFER_SIZE_MB': '1536', 'HCU_STEP5_DEEPEP_FUSED_SCATTER': '1', 'HCU_STEP5_DEEPEP_COMPACT_ROWS': '1', 'HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT': '1', 'HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR': '1', 'DEEP_EP_NORMAL_MNVL': '1', 'VLLM_USE_DEEP_GEMM': '1', 'VLLM_MOE_USE_DEEP_GEMM': '1', 'NCCL_MAX_NCHANNELS': '32'}
RATES = [0.838992, 0.598322, 0.362686]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_assets(model, assets):
    manifest = json.loads((assets / 'manifest.json').read_text())
    if sha(model / 'config.json') != manifest['model_config_sha256']:
        raise ValueError('This profile requires the delivered Step5 channel-FP8 checkpoint configuration')
    for relative, expected in manifest['assets'].items():
        path = (assets / relative).resolve()
        if not path.is_relative_to(assets.resolve()) or sha(path) != expected:
            raise ValueError('Asset mismatch: ' + relative)
    plan = json.loads((assets / 'expert_placement_prefill_e23_static_tokenhash_20261023.json').read_text())
    rows = plan['model_maps']['Step5ForCausalLM']['physical_to_logical_map']
    if len(rows) != 88 or any(len(row) != 368 or set(row) != set(range(352)) for row in rows):
        raise ValueError('Expected88 layers,352 logical experts and368 fixed physical experts')
    return manifest


def validate_install(manifest):
    import torch
    import deep_ep
    import vllm_hcu
    import vllm_hcu.hcu_ops
    for name, module in [('deep_ep', deep_ep), ('vllm_hcu', vllm_hcu)]:
        installed = metadata.distribution(name).locate_file(name + '/__init__.py')
        if Path(module.__file__).resolve() != Path(installed).resolve():
            raise ValueError(name + ' must be imported from its installed wheel')
    cpp = Path(deep_ep.__file__).parent / ('deep_ep_cpp' + __import__('sysconfig').get_config_var('EXT_SUFFIX'))
    if sha(cpp) != manifest['deep_ep_cpp_sha256']:
        raise ValueError('DeepEP extension differs from the delivered MNVL build')
    if not hasattr(deep_ep.Buffer, 'dispatch_scatter') or not hasattr(torch.ops.hcu_ops, 'step5_fp8_dense_gemm'):
        raise ValueError('Required normal DeepEP/HCU operators are unavailable')
    if torch.cuda.device_count() != 4:
        raise ValueError('This profile requires exactly four visible local GPUs')
    for index in range(4):
        arch = torch.cuda.get_device_properties(index).gcnArchName.split(':')[0]
        if arch != 'gfx938':
            raise ValueError('This calibrated profile requires gfx938')
    config = deep_ep.Config(16, 8, 512, 16, 128)
    if config.get_nvl_buffer_size_hint(4096 * 2, 16) > 1536 * 1024**2:
        raise ValueError('Communication configuration exceeds the reserved buffer')
    return dict(python=sys.executable, plugin=str(Path(vllm_hcu.__file__).resolve()),
                deep_ep=str(Path(deep_ep.__file__).resolve()),
                versions={n: metadata.version(n) for n in ['vllm_hcu','deep_ep','torch','vllm']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=range(4), required=True)
    p.add_argument('--head-ip', required=True)
    p.add_argument('--node-ip', required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('/var/cache/step5-release'))
    p.add_argument('--model', type=Path, default=Path('/data/model/Step5-Channel-FP8-w8a8'))
    p.add_argument('--iface', default='eno2')
    p.add_argument('--port', type=int, default=8018)
    p.add_argument('--rpc-port', type=int, default=18192)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    ipaddress.ip_address(a.head_ip); ipaddress.ip_address(a.node_ip)
    if not Path('/sys/class/net', a.iface).is_dir() or not (1 <= a.port <= 65535 and 1 <= a.rpc_port <= 65535):
        raise ValueError('Invalid interface or port')
    assets, state, cache, model = [x.resolve() for x in [a.assets,a.state,a.cache,a.model]]
    state.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    # Execute from run state, so a source checkout cannot shadow installed modules.
    os.chdir(state)
    for key in list(os.environ):
        if key.startswith(('HCU_STEP5_', 'VLLM_STEP5_', 'VLLM_STEP4_', 'HCU_DP_MOE_', 'HCU_DBO_', 'HCU_STATIC_EPLB_')):
            os.environ.pop(key)
    for key in ['PYTHONPATH','LD_PRELOAD','LD_AUDIT','HIPBLASLT_TUNING_OVERRIDE_FILE','ROCBLAS_TENSILE_LIBPATH','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','NCCL_TOPO_FILE','NCCL_IB_DISABLE','NCCL_MIN_NCHANNELS','VLLM_PLUGINS','VLLM_HCU_DP_PACKED_AG','HCU_EP_DIRECT_SUM']:
        os.environ.pop(key, None)
    os.environ.update(ENVIRONMENT)
    os.environ.update(PYTHONNOUSERSITE='1',VLLM_HOST_IP=a.node_ip,NCCL_SOCKET_IFNAME=a.iface,GLOO_SOCKET_IFNAME=a.iface,
                      HOME=str(state/'home'),TMPDIR=str(cache/'tmp'),XDG_CACHE_HOME=str(cache),HF_HOME=str(cache/'huggingface'),
                      VLLM_CACHE_ROOT=str(cache/'vllm'),TORCHINDUCTOR_CACHE_DIR=str(cache/'torchinductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                      CUDA_CACHE_PATH=str(cache/'cuda'),NUMBA_CACHE_DIR=str(cache/'numba'),MIOPEN_USER_DB_PATH=str(cache/'miopen'),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache/'miopen'),PYTHONPYCACHEPREFIX=str(cache/'pycache'),
                      HCU_STEP5_TUNING_CONFIG=str(assets/'tuning/step5-gfx938-image1021.json'),
                      VLLM_STEP5_EXPERT_PLACEMENT=str(assets/'expert_placement_prefill_8192_ep16_20261020.pt'))
    for key in ['HOME','TMPDIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','MIOPEN_USER_DB_PATH']:
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    if len(os.environ['TMPDIR'].encode()) > 65:
        raise ValueError('Cache path is too long for ZMQ Unix sockets')
    manifest = validate_assets(model, assets)
    receipt = validate_install(manifest)
    profile = dict(version=1,num_sms=16,dispatch_send=8,combine_send=8,recv_tokens=512,async_layout=True,overlap=True)
    runtime = state/'communication.json'
    os.environ['HCU_STEP5_DEEPEP_RUNTIME_CONFIG'] = str(runtime)
    eplb = dict(num_redundant_experts=16,use_async=False,disable_rearrange=True,
                expert_map_path=str(assets/'expert_placement_prefill_e23_static_tokenhash_20261023.json'),static_dispatch_policy='token_hash')
    command=[sys.executable,'-u','-m','vllm.entrypoints.cli.main','serve',str(model),
      '--tokenizer',str(assets/'tokenizer-bytelevel'),'--served-model-name','step5','--host','127.0.0.1','--port',str(a.port),
      '--tensor-parallel-size','1','--data-parallel-size','16','--data-parallel-size-local','4',
      '--data-parallel-address',a.head_ip,'--data-parallel-rpc-port',str(a.rpc_port),'--distributed-executor-backend','mp',
      '--enable-expert-parallel','--all2all-backend','deepep_high_throughput','--moe-backend','auto',
      '--max-model-len','70000','--max-num-seqs','64','--max-num-batched-tokens','8192','--long-prefill-token-threshold','0',
      '--kv-cache-dtype','fp8_e4m3','--gpu-memory-utilization','0.875','--no-enable-prefix-caching','--max-cudagraph-capture-size','64',
      '--compilation-config',json.dumps(dict(cudagraph_mode='FULL_AND_PIECEWISE',custom_ops=['+swiglustep_and_mul'],compile_sizes=[16,64])),
      '--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=3,rejection_sample_method='synthetic',synthetic_acceptance_rates=RATES,use_local_argmax_reduction=True)),
      '--limit-mm-per-prompt','{"image":0}','--enable-eplb','--eplb-config',json.dumps(eplb)]
    command += ['--api-server-count','1'] if a.rank == 0 else ['--headless','--data-parallel-start-rank',str(a.rank*4)]
    receipt.update(argv=command,rank=a.rank,profile=profile,asset_manifest_sha256=sha(assets/'manifest.json'))
    print(json.dumps(receipt),flush=True)
    if a.dry_run:
        return
    with (state/'launch.json').open('x') as f:json.dump(receipt,f,indent=2)
    runtime.write_text(json.dumps(profile))
    os.execv(sys.executable,command)


if __name__ == '__main__':
    main()
