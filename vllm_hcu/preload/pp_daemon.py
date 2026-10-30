# SPDX-License-Identifier: Apache-2.0
"""Pipeline-aware weight-cache daemon launcher.

The vLLM 0.28 daemon predates PP support and starts a TP-only process group.
This module keeps the installed protocol and model loader intact while
launching one cache process for every global TP/PP rank.  Each process loads
the stage selected by vLLM's normal PP group and serves a UUID+stage socket.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait
import os
import queue
import signal


def _run_pipeline_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue,
    tp_size: int,
    pp_size: int,
) -> None:
    """Load one PP/TP stage and keep its CUDA tensors available."""
    os.environ["VLLM_HCU_PRELOAD_PP_SIZE"] = str(pp_size)
    os.environ["VLLM_HCU_PRELOAD_PP_RANK"] = str(global_rank // tp_size)

    # Each spawned process must arm the same plugin callbacks as an engine
    # worker before importing the installed daemon/model loader modules.
    from vllm_hcu.preload.daemon import _prepare

    daemon_module = _prepare()
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.platforms import current_platform

    class PipelineWeightCacheDaemon(daemon_module.WeightCacheDaemon):
        def __init__(self):
            super().__init__(
                vllm_config,
                global_rank % tp_size,
                distributed_init_method,
                socket_dir,
            )
            self.global_rank = global_rank
            self.local_rank = local_rank

        def load_model(self) -> None:
            torch.accelerator.set_device_index(self.local_rank)
            init_distributed_environment(
                world_size=tp_size * pp_size,
                rank=self.global_rank,
                distributed_init_method=distributed_init_method,
                local_rank=self.local_rank,
                backend=current_platform.dist_backend,
            )
            with set_current_vllm_config(vllm_config):
                ensure_model_parallel_initialized(
                    tp_size,
                    pp_size,
                    backend=current_platform.dist_backend,
                )
                self.model = get_model(vllm_config=vllm_config)
            self._export_entries()

    daemon = PipelineWeightCacheDaemon()
    daemon.load_model()
    daemon.serve_forever(ready_callback=lambda: ready_queue.put(global_rank))


def main(daemon_module) -> None:
    """Parse full-model daemon arguments and launch a PP-aware local group."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_distributed_init_method, get_open_port

    parser = FlexibleArgumentParser(
        description="Launch pipeline-aware weight cache daemons (one per TP/PP rank)."
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--weight-cache-socket-dir", type=str, default=None)
    parser.add_argument(
        "--weight-cache-master-port",
        type=int,
        default=None,
        help="Rendezvous port for the daemon TP/PP group; use a port distinct "
        "from the engine master port.",
    )
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    parallel = vllm_config.parallel_config
    tp_size = int(parallel.tensor_parallel_size)
    pp_size = int(parallel.pipeline_parallel_size)
    if pp_size <= 1:
        # This path is retained for the existing, installed daemon behavior.
        return daemon_module.main()
    if vllm_config.load_config.load_format == "ipc_cache":
        raise ValueError(
            "the weight cache daemon must load from disk; do not pass "
            "--load-format ipc_cache to the daemon"
        )
    if parallel.data_parallel_size != 1:
        raise ValueError("pipeline weight cache currently requires data_parallel_size=1")
    if parallel.enable_expert_parallel:
        raise ValueError("pipeline weight cache currently does not support expert parallelism")

    daemon_module.check_ipc_quant_support(vllm_config.model_config, where="daemon")
    nnodes = int(parallel.nnodes)
    node_rank = int(parallel.node_rank)
    world_size = tp_size * pp_size
    if not 0 <= node_rank < nnodes:
        raise ValueError(f"--node-rank must be in [0, {nnodes}), got {node_rank}")
    if world_size % nnodes:
        raise ValueError(
            f"TP*PP world size ({world_size}) must be divisible by --nnodes ({nnodes})"
        )
    local_world_size = world_size // nnodes
    if parallel.master_addr is None:
        raise ValueError("--master-addr is required for pipeline weight cache")
    if nnodes > 1 and args.weight_cache_master_port is None:
        raise ValueError("--weight-cache-master-port is required when --nnodes > 1")
    master_port = args.weight_cache_master_port or get_open_port()
    init_method = get_distributed_init_method(parallel.master_addr, master_port)

    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    processes = []
    for global_rank, local_rank, _, _ in plan_local_pipeline_ranks(
        tp_size, pp_size, nnodes, node_rank
    ):
        process = context.Process(
            target=_run_pipeline_daemon,
            args=(
                global_rank,
                local_rank,
                vllm_config,
                init_method,
                args.weight_cache_socket_dir,
                ready_queue,
                tp_size,
                pp_size,
            ),
            name=f"vllm-weight-cache-daemon-pp{global_rank}",
        )
        process.start()
        processes.append(process)

    stopping = False

    def shutdown(_signum, _frame):
        nonlocal stopping
        stopping = True
        for process in processes:
            process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if not _wait_until_ready(processes, ready_queue, local_world_size, lambda: stopping):
        return
    print(
        f"Weight cache daemon READY: node {node_rank}/{nnodes}, "
        f"{local_world_size} local TP/PP ranks, PP={pp_size}",
        flush=True,
    )
    _monitor_processes(processes, lambda: stopping)


def _wait_until_ready(processes, ready_queue, local_world_size, is_stopping):
    ready = set()
    while len(ready) < local_world_size:
        if is_stopping():
            break
        try:
            ready.add(ready_queue.get(timeout=1.0))
        except queue.Empty:
            if is_stopping():
                break
            dead = [process for process in processes if process.exitcode is not None]
            if dead:
                for process in processes:
                    process.terminate()
                for process in processes:
                    process.join()
                raise SystemExit(max(process.exitcode or 1 for process in dead))
    if is_stopping():
        for process in processes:
            process.join()
        return False
    return True


def _monitor_processes(processes, is_stopping):
    """Stop the local group when any indefinitely serving child exits."""
    ready = set(wait([process.sentinel for process in processes]))
    failed = [process for process in processes if process.sentinel in ready]
    for process in failed:
        process.join()
    for process in processes:
        if process.exitcode is None:
            process.terminate()
    for process in processes:
        process.join()
    if not is_stopping():
        names = ", ".join(f"{p.name} (exit={p.exitcode})" for p in failed)
        raise SystemExit(f"Weight cache daemon exited unexpectedly: {names}")


def plan_local_pipeline_ranks(
    tp_size: int,
    pp_size: int,
    nnodes: int,
    node_rank: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(global, local, pp, tp)`` ranks for one node."""
    if tp_size < 1 or pp_size < 1 or nnodes < 1:
        raise ValueError("TP, PP and node count must be positive")
    world_size = tp_size * pp_size
    if not 0 <= node_rank < nnodes:
        raise ValueError(f"node rank {node_rank} is outside [0, {nnodes})")
    if world_size % nnodes:
        raise ValueError("TP*PP world size must be divisible by node count")
    local_world_size = world_size // nnodes
    first_global_rank = node_rank * local_world_size
    return [
        (
            global_rank,
            local_rank,
            global_rank // tp_size,
            global_rank % tp_size,
        )
        for local_rank, global_rank in enumerate(
            range(first_global_rank, first_global_rank + local_world_size)
        )
    ]
