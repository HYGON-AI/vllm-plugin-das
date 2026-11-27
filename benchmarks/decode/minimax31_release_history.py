"""Drain GPU work and serialize task-owned USERPTR teardown across DP ranks."""
import json
import torch


def release(*,runtime,rank,world):
    from hcu_vllm_das.parallel.minimax31_hisparse import _release_host_region,_unregister
    histories=[runtime.backing]
    bootstrap=getattr(runtime,'bootstrap_history_backing',None)
    if bootstrap is not None and bootstrap is not runtime.backing:histories.append(bootstrap)
    torch.cuda.synchronize();world.barrier()
    for turn in range(8):
        if turn==rank:
            print('MINIMAX_HISTORY_RELEASE_BEGIN '+json.dumps({'rank':rank}),flush=True)
            for backing in histories:
                owned=backing._finalizer.peek()
                if owned is None or owned[1] is not _release_host_region:raise RuntimeError('unknown history lifetime during release')
                native,ranges,region=owned[2]
                if region is not None or tuple(ranges)!=backing.ranges:raise RuntimeError('history registration owner changed')
                ranges[:]=_unregister(native,ranges);backing.ranges=tuple(ranges)
                if ranges:raise RuntimeError('owned USERPTR release failed')
                backing._finalizer.detach();backing.registered_bytes=0
            print('MINIMAX_HISTORY_RELEASE_END '+json.dumps({'rank':rank}),flush=True)
        world.barrier()
