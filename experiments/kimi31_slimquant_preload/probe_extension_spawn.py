"""Explicit experimental adapter extension; parent and one spawned child."""
import json, multiprocessing as mp, os, traceback
from pathlib import Path
os.environ['VLLM_PLUGINS']='hcu_adapter'

def inspect(q):
    result={'pid':os.getpid(), 'ppid':os.getppid(), 'hostname':os.uname().nodename}
    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        from hcu_adapter.bridges.vllm import require_runtime_healthy, extension_reports
        from hcu_adapter.bridges.vllm_kimi31_preload import selected_mode
        from vllm.model_executor.layers.quantization import get_quantization_config
        from vllm.model_executor.models.registry import ModelRegistry
        from hcu_vllm_das.quantization.preload_activation import check_health
        from hcu_vllm_das.quantization.kimi31_load_model import OUTER_ARCH,INNER_ARCH
        check_health(); require_runtime_healthy()
        result['mode']=selected_mode()
        result['quant']=get_quantization_config('slimquant_w4a8').__qualname__
        result['outer']=ModelRegistry.models[OUTER_ARCH].model_cls.__qualname__
        result['inner']=ModelRegistry.models[INNER_ARCH].model_cls.__qualname__
        result['reports']=[{'entry':r.entry_point_name,'status':str(r.status),'capabilities':sorted(r.capabilities)} for r in extension_reports()]
        result['ok']=True
    except BaseException:
        result['ok']=False; result['traceback']=traceback.format_exc()
    q.put(result)

if __name__ == '__main__':
    ctx=mp.get_context('spawn'); q=ctx.Queue()
    inspect(q); parent=q.get(timeout=60)
    child=ctx.Process(target=inspect,args=(q,)); child.start(); child_result=q.get(timeout=180); child.join(180)
    result={'parent':parent,'child':child_result,'child_exit':child.exitcode,
            'scope':'plugin bootstrap only; no model constructed, weights loaded, or IPC'}
    Path(os.environ.get("K31_PROBE_OUTPUT", str(Path(__file__).with_suffix(".json")))).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not parent['ok'] or not child_result['ok'] or child.exitcode != 0: raise SystemExit(1)
