"""Installed communication contract, not model/custom-SP qualification."""
from datetime import timedelta
from pathlib import Path
import hashlib
import inspect
import json
import sys
import traceback

OUT = Path(sys.argv[1])
ENABLED = sys.argv[2] == "candidate"
PREFIX = Path(sys.prefix).resolve()


def worker(rank):
    result = dict(rank=rank, enabled=ENABLED, cases=[])
    import torch
    torch.cuda.set_device(rank)
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.distributed import (
        init_distributed_environment, initialize_model_parallel,
        cleanup_dist_env_and_memory, get_tp_group, get_ep_group,
    )
    from vllm.model_executor.layers.fused_moe.layer import make_parallel_config
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from hcu_adapter.bridges.vllm import require_extension_healthy
    require_extension_healthy("runtime-sp-communication")
    result["architecture"] = torch.cuda.get_device_properties(rank).gcnArchName
    parallel = ParallelConfig(tensor_parallel_size=2, enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter", disable_custom_all_reduce=True)
    config = VllmConfig(parallel_config=parallel, device_config=DeviceConfig("cuda"),
        additional_config={"hcu": {"enable_custom_sp": ENABLED}})
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(world_size=2, rank=rank, local_rank=rank,
                distributed_init_method="file://" + str(OUT / "rendezvous"),
                backend="nccl", timeout=timedelta(seconds=90))
            initialize_model_parallel(2, 1)
            tp, ep = get_tp_group(), get_ep_group()
            moe = make_parallel_config(2, 1, 1, True, parallel)
            comm = ep.device_communicator
            result["construction"] = dict(
                selected=current_platform.get_device_communicator_cls(),
                parallel_use_all2all=parallel.use_all2all,
                moe_use_all2all=moe.use_all2all_kernels,
                communicator_use_all2all=comm.use_all2all,
                manager=None if comm.all2all_manager is None else type(comm.all2all_manager).__name__,
                tp_communicator=type(tp.device_communicator).__module__ + "." + type(tp.device_communicator).__name__,
            )
            assert not parallel.use_all2all and moe.use_all2all_kernels
            if not ENABLED:
                assert comm.all2all_manager is None
                try:
                    ep.dispatch_router_logits(torch.ones(3, 8, device="cuda"),
                        torch.ones(3, 4, device="cuda"), True)
                except AssertionError:
                    result["baseline_missing_manager_reproduced"] = traceback.format_exc()
                else:
                    raise AssertionError("baseline unexpectedly dispatched without a manager")
            else:
                assert type(comm.all2all_manager).__name__ == "AgRsAll2AllManager"
                assert not tp.device_communicator.use_all2all
                for rows, width, dtype in [(3, 8, torch.float32), (5, 16, torch.bfloat16)]:
                    h = torch.empty(rows, width, device="cuda", dtype=dtype)
                    logits = torch.empty(rows, 4, device="cuda", dtype=dtype)
                    weights = torch.empty(rows, 2, device="cuda", dtype=dtype)
                    ids = torch.empty(rows, 2, device="cuda", dtype=torch.int64)
                    extra = torch.empty(rows, 3, device="cuda", dtype=torch.uint8)
                    combined_out = torch.empty_like(h)
                    heads = torch.empty(rows, 4, 8, device="cuda", dtype=dtype)
                    send = torch.empty(2, rows, 2, 8, device="cuda", dtype=dtype)
                    recv = torch.empty_like(send)
                    inverse = torch.empty_like(send)
                    restored = torch.empty_like(heads)

                    def full(epoch):
                        def values(columns, mod, shift=0):
                            raw = torch.arange(2 * rows * columns).reshape(2 * rows, columns)
                            return ((raw + epoch + shift) % mod - mod // 2).to(dtype) / 8
                        return [values(width, 17), values(4, 19), values(2, 7),
                            (torch.arange(4 * rows).reshape(2 * rows, 2) + epoch) % 4,
                            ((torch.arange(6 * rows).reshape(2 * rows, 3) + epoch) % 251).byte(),
                            values(32, 23).reshape(2 * rows, 4, 8)]

                    def update(epoch):
                        for local, global_value in zip((h, logits, weights, ids, extra, heads), full(epoch)):
                            local.copy_(global_value[rank * rows:(rank + 1) * rows])

                    def execute():
                        router = ep.dispatch_router_logits(h, logits, True, [extra])
                        dispatched = ep.dispatch(h, weights, ids, True, [extra])
                        combined = ep.combine(router[0] * (rank + 1), True)
                        into = comm.combine_into_output(dispatched[0] * (rank + 1), combined_out, True)
                        assert into.data_ptr() == combined_out.data_ptr()
                        assert into.shape == combined_out.shape and into.dtype == combined_out.dtype
                        assert into.device == combined_out.device
                        send.copy_(heads.view(rows, 2, 2, 8).permute(1, 0, 2, 3))
                        assert tp.device_communicator.all_to_all_single(recv, send) is recv
                        tp.device_communicator.all_to_all_single(inverse, recv)
                        restored.copy_(inverse.permute(1, 0, 2, 3).reshape_as(heads))
                        return [router[0], router[1], router[2][0], dispatched[0],
                            dispatched[1], dispatched[2], dispatched[3][0],
                            combined, combined_out, recv, restored]

                    def validate(outputs, epoch):
                        base = full(epoch)
                        expected = [base[0], base[1], base[4], base[0], base[2],
                            base[3], base[4], base[0][rank * rows:(rank + 1) * rows] * 3,
                            base[0][rank * rows:(rank + 1) * rows] * 3,
                            base[5][:, rank * 2:(rank + 1) * 2].reshape_as(recv.cpu()),
                            base[5][rank * rows:(rank + 1) * rows]]
                        for actual, target in zip(outputs, expected):
                            torch.testing.assert_close(actual.cpu(), target, rtol=0, atol=0)

                    update(0)
                    for _ in range(3):
                        outputs = execute()
                    torch.cuda.synchronize()
                    validate(outputs, 0)
                    graph = torch.cuda.CUDAGraph()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    dist.barrier(group=tp.device_group)
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = execute()
                    torch.cuda.current_stream().wait_stream(stream)
                    pointers = [value.data_ptr() for value in outputs]
                    for epoch in (1, 5):
                        update(epoch)
                        graph.replay()
                        torch.cuda.synchronize()
                        validate(outputs, epoch)
                        assert pointers == [value.data_ptr() for value in outputs]
                    graph.reset()
                    del graph, outputs
                    dist.barrier(group=tp.device_group)
                    result["cases"].append(dict(rows=rows, width=width, dtype=str(dtype),
                        dispatch_router_logits=True, dispatch_topk=True, extra_uint8=True,
                        combine=True, combine_into_output=True, token_head_exchange=True,
                        inverse=True, eager=True, changed_input_graph_epochs=[1, 5],
                        zero_tolerance=True, addresses_stable=True))
            result["origins"] = {}
            for name, owner in [("communicator", type(comm)), ("moe_factory", make_parallel_config)]:
                path = Path(inspect.getfile(owner)).resolve()
                assert path.is_relative_to(PREFIX), str(path)
                result["origins"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            result["success"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (OUT / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(worker, nprocs=2, join=True)
