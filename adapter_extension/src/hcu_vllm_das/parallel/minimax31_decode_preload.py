# SPDX-License-Identifier: Apache-2.0
"""Single-node DP8 weight owners for explicit MiniMax decode experiments."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import queue
import signal
import time
import uuid


def draft_config(config):
    """Use the exact V2 DSpark effective-config rules, without its allowlist."""
    from vllm.config import replace
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
        _get_dspark_parallel_config,_resolve_dspark_attention_backend)
    spec = config.speculative_config
    model = spec.draft_model_config
    result = replace(config,
        parallel_config=_get_dspark_parallel_config(config.parallel_config,
            spec.draft_parallel_config.tensor_parallel_size,model),
        attention_config=replace(config.attention_config,
            use_non_causal=dflash_has_any_non_causal(model.hf_config),
            backend=_resolve_dspark_attention_backend(model,spec.attention_backend,
                                                       config.attention_config.backend)),
        cache_config=replace(config.cache_config,cache_dtype=spec.kv_cache_dtype)
            if spec.kv_cache_dtype is not None else config.cache_config)
    result.quant_config = get_draft_quant_config(config)
    return result


def _worker(rank,role,config,address,socket_dir,generation,ready):
    import torch
    torch.accelerator.set_device_index(rank)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from .minimax31_decode_ipc import DecodeDaemon
    from .minimax31_decode_activation import check_health
    check_health()
    daemon = DecodeDaemon(config,rank,rank,address,socket_dir,
                         is_draft=role=="draft",dp_rank=rank)
    daemon.decode_generation = generation
    daemon.load_model()
    daemon.warmup()
    daemon.release_distributed()
    def publish():
        from vllm.model_executor.model_loader.weight_cache.daemon import exported_weight_bytes
        ready.put({"role":role,"dp_rank":rank,"pid":os.getpid(),
            "process_birth":Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            "pid_namespace":"container","gpu_uuid":daemon.decode_partition.gpu_uuid,
            "partition":daemon.decode_partition.to_dict(),"socket":daemon._socket_path,
            "weight_bytes":exported_weight_bytes(daemon.model)})
    daemon.serve_forever(ready_callback=publish)


def stop_workers(procs,grace=15):
    live = [p for p in procs if p.pid is not None]
    for p in live:
        if p.is_alive():p.terminate()
    deadline=time.monotonic()+grace
    for p in live:p.join(max(0,deadline-time.monotonic()))
    for p in live:
        if p.is_alive():p.kill()
    for p in live:p.join(5)


def check_peer_owners(rows):
    """A reused target is this container's live publication, never a stale PID."""
    for row in rows:
        pid=row.get('pid')
        if type(pid) is not int or row.get('pid_namespace')!='container':
            raise ValueError('reused target requires same-container owner receipts')
        try:birth=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
        except (OSError,IndexError) as exc:raise RuntimeError('reused target owner died') from exc
        if birth!=row.get('process_birth'):raise RuntimeError('reused target PID changed')


def main(argv=None):
    # Match Core's CLI startup order: platform/general owners must be active
    # before EngineArgs asks the platform to validate parser defaults.
    from vllm.entrypoints.common.startup import cli_env_setup
    cli_env_setup()
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.entrypoints.cli.preload import PreloadSubcommand
    parser=argparse.ArgumentParser(description=__doc__)
    PreloadSubcommand.add_cli_args(parser)
    parser.add_argument('--decode-ready-manifest',required=True)
    parser.add_argument('--decode-reuse-target-manifest',
        help='reuse eight live target owners and load only the independent DSpark draft')
    args=parser.parse_args(argv)
    if not args.weight_cache_socket_dir:
        parser.error('--weight-cache-socket-dir is required')
    root=Path(args.weight_cache_socket_dir).absolute()
    longest=root/('vllm_weight_cache_'+'0'*64+'_draft.sock')
    if root.is_symlink() or len(os.fsencode(longest))>=108:
        parser.error('socket directory must be a short real directory')
    destination=Path(args.decode_ready_manifest).absolute()
    if destination.exists() or not destination.parent.is_dir():
        parser.error('ready manifest must be new and its parent must exist')
    if any((args.seed_listen_host,args.seed_source_host,args.seed_listen_port,args.seed_source_port,args.seed_token)):
        parser.error('remote seed transport is outside this single-node contract')
    from vllm.engine.arg_utils import EngineArgs
    from vllm.plugins import load_general_plugins
    from vllm.utils.network_utils import get_distributed_init_method,get_open_port
    from .minimax31_decode import validate_scope
    from .minimax31_decode_contract import ready_roles
    from .minimax31_decode_ipc import weight_identity
    load_general_plugins()
    config=EngineArgs.from_cli_args(args).create_engine_config()
    validate_scope(config)
    if config.load_config.load_format!='safetensors':
        parser.error('initial decode producer requires ordinary safetensors')
    weight_identity(config.model_config)
    configs=[('target',config)]
    peer_rows=[]
    if config.speculative_config is not None:
        if not config.speculative_config.use_dspark():
            parser.error('only independent DSpark draft is supported')
        dc=draft_config(config)
        weight_identity(dc.speculative_config.draft_model_config)
        configs.append(('draft',dc))
    if args.decode_reuse_target_manifest:
        if len(configs)!=2:parser.error('target reuse requires an independent DSpark draft')
        peer=json.loads(Path(args.decode_reuse_target_manifest).read_text())
        if peer.get('schema')!='minimax31-decode-ready-v1' or peer.get('with_draft') is not False:
            parser.error('target reuse requires a target-only ready manifest')
        ready_roles(peer['ranks'],with_draft=False)
        peer_rows=[dict(row,generation=row.get('generation',peer['generation'])) for row in peer['ranks']]
        check_peer_owners(peer_rows)
        configs=configs[1:]
    ctx=mp.get_context('spawn');ready=ctx.Queue();generation=uuid.uuid4().hex
    procs=[]
    for role,cfg in configs:
        port=(args.weight_cache_master_port if role=='target' else args.weight_cache_draft_master_port) or get_open_port()
        address=get_distributed_init_method('127.0.0.1',port)
        procs += [ctx.Process(target=_worker,args=(r,role,cfg,address,str(root),generation,ready),
                            name='minimax31-decode-%s-%d'%(role,r)) for r in range(8)]
    stopping=False
    def shutdown(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    try:
        for p in procs:p.start()
        rows=[]
        while len(rows)<len(procs) and not stopping:
            try:rows.append(ready.get(timeout=1))
            except queue.Empty:pass
            failed=[(p.name,p.exitcode) for p in procs if p.exitcode is not None]
            if failed:raise RuntimeError('decode producer exited before READY: '+str(failed))
        if stopping:return 1
        rows=[dict(row,generation=generation) for row in rows]+peer_rows
        with_draft=any(row['role']=='draft' for row in rows)
        ready_roles(rows,with_draft=with_draft)
        record={'schema':'minimax31-decode-ready-v1','generation':generation,
                'with_draft':with_draft,'ranks':rows,
                'scope':'ready observation; live socket and owner watch remain required'}
        with destination.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print('MINIMAX_DECODE_PRELOAD_READY '+json.dumps(record),flush=True)
        while not stopping:
            check_peer_owners(peer_rows)
            exited=wait([p.sentinel for p in procs],timeout=1)
            if exited and not stopping:
                raise RuntimeError('decode owner died after READY: '+str(
                    [(p.name,p.exitcode) for p in procs if p.sentinel in exited]))
        return 0
    finally:
        stop_workers(procs)


if __name__=='__main__':
    raise SystemExit(main())
