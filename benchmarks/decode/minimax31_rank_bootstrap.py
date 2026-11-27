"""Bind node20 CPU/memory placement before sitecustomize, Torch or HSA loads."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import site
import sys

assert 'torch' not in sys.modules,'invoke this bootstrap with python -S'
rank=int(os.environ['LOCAL_RANK'])
if os.environ.get('M31_SINGLE_VISIBLE_DEVICE')=='1':
    visible=os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices=visible.split(',')
        if len(devices)!=8:raise RuntimeError('expected the original eight-device visibility for partitioning')
        physical=devices[rank]
    else:physical=str(rank)
    if os.environ.get('M31_SINGLE_VISIBLE_ROCR')=='1':
        if os.environ.get('ROCR_VISIBLE_DEVICES'):
            raise RuntimeError('ROCR partitioning requires an unmasked parent')
        os.environ['ROCR_VISIBLE_DEVICES']=physical
        physical='0'
    os.environ['HIP_VISIBLE_DEVICES']=physical
    os.environ['CUDA_VISIBLE_DEVICES']=physical
assignment=[int(x) for x in os.environ['M31_NUMA_ASSIGNMENT'].split(',')]
assert len(assignment)==8 and sorted(assignment)==list(range(8))
node=assignment[rank]
cpus=set()
for part in (Path('/sys/devices/system/node')/('node%d'%node)/'cpulist').read_text().strip().split(','):
    bounds=[int(x) for x in part.split('-')];cpus.update(range(bounds[0],bounds[-1]+1))
cpus &= os.sched_getaffinity(0)
if not cpus:raise RuntimeError('NUMA node outside allowed CPU affinity')
os.sched_setaffinity(0,cpus)
policy='cpu-only';errors=[]
try:
    numa=ctypes.CDLL('libnuma.so.1',use_errno=True)
    numa.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
    numa.set_mempolicy.restype=ctypes.c_int
    mask=ctypes.c_ulong(1<<node)
    for mode,name in ((2,'bind'),(1,'preferred')):
        ctypes.set_errno(0)
        if numa.set_mempolicy(mode,ctypes.byref(mask),ctypes.sizeof(mask)*8)==0:
            policy=name;break
        errors.append({'mode':name,'errno':ctypes.get_errno()})
except (OSError,AttributeError) as exc:errors.append({'exception':str(exc)})
print('MINIMAX_EARLY_NUMA '+json.dumps({'rank':rank,'node':node,'cpus':sorted(cpus),
    'memory_policy':policy,'policy_probe_errors':errors,
    'HSA_USERPTR_FOR_PAGED_MEM':os.environ.get('HSA_USERPTR_FOR_PAGED_MEM'),
    'HIP_VISIBLE_DEVICES':os.environ.get('HIP_VISIBLE_DEVICES'),
    'ROCR_VISIBLE_DEVICES':os.environ.get('ROCR_VISIBLE_DEVICES'),
    'torch_imported_before_binding':False}),flush=True)
if os.environ.get('M31_HIP_DEBUG_DIR'):
    os.environ['HIP_LOG_LEVEL']='4'
    os.environ['HIP_LOG_OUTPUT_PATH']=str(Path(os.environ['M31_HIP_DEBUG_DIR'])/('rank%d.bin'%rank))
site.main()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0],run_name='__main__')
