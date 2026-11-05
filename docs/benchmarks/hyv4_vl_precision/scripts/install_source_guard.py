import ast
import textwrap
import types
import vllm_hcu.v1.hcu_model_runner as runner_module
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.forward_context import is_forward_context_available,get_forward_context
from collections import Counter
path='/tmp/vllm-plugin-das-pr/vllm_hcu/v1/hcu_model_runner.py'
source=open(path).read()
tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_determine_batch_execution_and_padding')
module=ast.Module(body=[fn],type_ignores=[])
namespace={}
exec(compile(ast.fix_missing_locations(module),path,'exec'),runner_module.__dict__,namespace)
runner_module.GPUModelRunner._determine_batch_execution_and_padding=namespace[fn.name]
worker._stage4_actual=Counter()
worker._stage4_actual_shapes=Counter()
original=BreakableCUDAGraphWrapper.__call__
def call(wrapper,*args,**kwargs):
    if is_forward_context_available():
        ctx=get_forward_context()
        worker._stage4_actual[str(ctx.cudagraph_runtime_mode)]+=1
        worker._stage4_actual_shapes[str((ctx.cudagraph_runtime_mode,ctx.batch_descriptor))]+=1
    return original(wrapper,*args,**kwargs)
BreakableCUDAGraphWrapper.__call__=call
result={'rank':worker.rank,'source_path':path,'lines':[fn.lineno,fn.end_lineno]}
