# SPDX-License-Identifier: Apache-2.0
"""One-node PCP8/EP8 preload using guarded native vLLM IPC primitives."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import sys

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.preload import PreloadSubcommand, _HealthState, _start_health_server
from vllm.model_executor.model_loader.weight_cache.protocol import check_ipc_platform_support
from vllm.plugins import load_general_plugins
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

from .minimax31_ipc import MiniMax31PcpEpDaemon, require_core_contract
from .minimax31_preload import plan_ranks, validate_parallel_config, require_extensions


def _worker(rank, config, address, socket_dir, ready_queue):
    # Spawn starts a fresh interpreter: activate both explicitly required
    # extensions in every producer before resolving the model class.
    load_general_plugins()
    require_extensions()
    daemon = MiniMax31PcpEpDaemon(config, rank, rank, address, socket_dir)
    daemon.load_model()
    daemon.warmup()
    from .minimax31_static_ipc import freeze_publication, publication_identity
    freeze_publication(daemon)
    daemon.release_distributed()
    daemon.serve_forever(ready_callback=lambda: ready_queue.put({
        "rank": rank, "pid": os.getpid(), "gpu_uuid": daemon._partition.gpu_uuid,
        "socket_path": daemon._socket_path,
        "static_publication": publication_identity(daemon),
    }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    PreloadSubcommand.add_cli_args(parser)
    parser.add_argument("--pcp-ep-ready-manifest", required=True)
    args = parser.parse_args(argv)
    if not args.weight_cache_socket_dir:
        parser.error("--weight-cache-socket-dir is required")
    if args.seed_listen_host or args.seed_listen_port or args.seed_source_host or args.seed_source_port or args.seed_token:
        parser.error("remote seed transport is outside the PCP8/EP8 contract")
    root = Path(args.weight_cache_socket_dir).absolute()
    if root.is_symlink() or len(os.fsencode(str(root / ("vllm_weight_cache_" + "0"*64 + ".sock")))) >= 108:
        parser.error("unsafe or too-long socket path")
    destination=Path(args.pcp_ep_ready_manifest).absolute()
    if destination.exists() or not destination.parent.is_dir():
        parser.error("ready manifest must not exist and its parent must already exist")
    load_general_plugins()
    require_core_contract()
    require_extensions()
    config=EngineArgs.from_cli_args(args).create_engine_config()
    validate_parallel_config(config.parallel_config)
    from .minimax31_pcp import validate_scope
    validate_scope(config)
    if config.load_config.load_format != "safetensors" or config.speculative_config is not None:
        parser.error("producer requires safetensors with no speculative model")
    check_ipc_platform_support()
    from .minimax31_static_contract import from_model_config
    placements=plan_ranks(config.parallel_config,from_model_config(config.model_config))
    address=get_distributed_init_method("127.0.0.1", args.weight_cache_master_port or get_open_port())
    ctx=multiprocessing.get_context("spawn")
    ready=ctx.Queue()
    procs=[ctx.Process(target=_worker,args=(row["global_rank"],config,address,str(root),ready),
                       name="minimax31-pcp-ep-%d" % row["global_rank"]) for row in placements]
    state=_HealthState(procs)
    server=thread=None
    if args.weight_cache_health_port is not None:
        server,thread=_start_health_server(args.weight_cache_health_host,args.weight_cache_health_port,state)
    stopping=False
    def shutdown(signum, frame):
        nonlocal stopping
        stopping=True
        if server is not None: server.should_exit=True
        for proc in procs:
            if proc.is_alive(): proc.terminate()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    try:
        for proc in procs: proc.start()
        observed={}
        while len(observed)<8 and not stopping:
            try: item=ready.get(timeout=1)
            except queue.Empty:
                failures=[p for p in procs if p.exitcode is not None]
                if failures:
                    raise RuntimeError("PCP/EP producer exited before readiness: %s" % [(p.name,p.exitcode) for p in failures])
                continue
            rank=item.get("rank")
            if rank not in range(8) or rank in observed:
                raise RuntimeError("invalid or duplicate producer readiness rank")
            observed[rank]=item
        if stopping: return 1
        uuids=[item["gpu_uuid"] for item in observed.values()]
        if len(set(uuids))!=8:
            raise RuntimeError("PCP/EP ranks must each own a distinct actual GPU UUID")
        record={"kind":"minimax31_pcp_ep_producer_ready_observation",
                "world_size":8,"ranks":[observed[r] for r in range(8)],
                "note":"historical readiness observation; query live sockets for current state"}
        with destination.open("x",encoding="utf8") as stream:
            json.dump(record,stream,ensure_ascii=False,indent=2);stream.write("\n")
        state.mark_ready()
        print("PCP8 EP8 preload READY: 8 local producers",flush=True)
        for proc in procs: proc.join()
        return max((proc.exitcode or 0) for proc in procs)
    finally:
        if server is not None: server.should_exit=True
        for proc in procs:
            if proc.is_alive(): proc.terminate()
        for proc in procs:
            if proc.pid is not None: proc.join()
        if thread is not None: thread.join()


if __name__ == "__main__":
    sys.exit(main())
