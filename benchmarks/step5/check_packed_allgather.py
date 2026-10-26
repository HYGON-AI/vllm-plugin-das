# SPDX-License-Identifier: Apache-2.0
"""Four-GPU transport check: torchrun --standalone --nproc-per-node=4 FILE."""
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

torch.set_num_threads(1)
rank = int(os.environ['RANK'])
local = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local)
dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator

patch_cuda_communicator.apply_to_module(cuda_communicator)
comm = object.__new__(cuda_communicator.CudaCommunicator)
comm.world_size = dist.get_world_size()
comm.rank_in_group = rank
comm._hcu_ipc_ag = None
comm.pynccl_comm = PyNcclCommunicator(dist.group.WORLD, torch.device('cuda', local))
torch.manual_seed(6479 + rank)
inputs = [torch.randn(64, 4096, device='cuda').to(torch.float8_e4m3fn),
          torch.rand(64, 8, device='cuda'),
          torch.randint(0, 352, (64, 8), device='cuda', dtype=torch.int32),
          torch.rand(64, 1, device='cuda')]
native = comm.pynccl_comm.all_gather
calls = []

def tracked(*args, **kwargs):
    calls.append(args[1].dtype)
    return native(*args, **kwargs)

comm.pynccl_comm.all_gather = tracked
comm._hcu_dp_packed_ag = False
reference = comm.all_gatherv(inputs, sizes=[64] * comm.world_size)
assert len(calls) == 4
calls.clear()
comm._hcu_dp_packed_ag = True
outputs = comm.all_gatherv(inputs, sizes=[64] * comm.world_size)
torch.cuda.synchronize()
assert calls == [torch.uint8]
assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
           for a, b in zip(reference, outputs))
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outputs = comm.all_gatherv(inputs, sizes=[64] * comm.world_size)
for value in [0, 1]:
    inputs[0].fill_(value + rank)
    for output in outputs:
        output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    comm._hcu_dp_packed_ag = False
    reference = comm.all_gatherv(inputs, sizes=[64] * comm.world_size)
    torch.cuda.synchronize()
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
               for a, b in zip(reference, outputs))
    comm._hcu_dp_packed_ag = True
result = dict(rank=rank,world=comm.world_size,bitwise_equal=True,
              collectives_before=4,collectives_after=1,
              changed_inputs_zeroed_outputs_graph_replay=True)
root = Path(os.environ.get('STEP5_PROOF_DIR', '/var/lib/xuwq_step5_native6479_review'))
root.mkdir(parents=True,exist_ok=True)
(root/f'packed-native-rank{rank}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
dist.barrier()
del graph
comm.pynccl_comm.destroy()
dist.destroy_process_group()
