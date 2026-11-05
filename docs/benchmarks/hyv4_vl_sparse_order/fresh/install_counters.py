from collections import Counter
import hashlib
from pathlib import Path
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.forward_context import get_forward_context,is_forward_context_available
from vllm_hcu.models.hy_v4.attention import Indexer
indexers=[m for m in worker.model_runner.model.modules() if isinstance(m,Indexer)]
assert len(indexers)==21
assert all(m.canonicalize_topk for m in indexers)
worker._fix_actual=Counter()
worker._fix_actual_shapes=Counter()
original=BreakableCUDAGraphWrapper.__call__
def call(wrapper,*args,**kwargs):
    if is_forward_context_available():
        ctx=get_forward_context()
        worker._fix_actual[str(ctx.cudagraph_runtime_mode)]+=1
        worker._fix_actual_shapes[str((ctx.cudagraph_runtime_mode,ctx.batch_descriptor))]+=1
    return original(wrapper,*args,**kwargs)
BreakableCUDAGraphWrapper.__call__=call
result={'rank':worker.rank,'enabled_indexers':len(indexers),'architectures':list(indexers[0].vllm_config.model_config.hf_config.architectures),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/tmp/vllm-plugin-das-pr/vllm_hcu/models/hy_v4/attention.py'),Path('/tmp/vllm-plugin-das-pr/vllm_hcu/models/hy_v4/sparse_indices.py')]}}
