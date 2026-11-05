import sys
sys.path.insert(0,'/root/hy4-perf-stage4')
from repro import rpc,mode,save,r
import importlib.util
from pathlib import Path
import pyarrow.parquet as pq
from types import SimpleNamespace
sys.path.insert(0,'/tmp/vllm-plugin-das-pr')
from tools.hyv4_realworldqa_eval import evaluate
rows=pq.read_table('/root/hy4-realworldqa/data/data/test-00000-of-00002.parquet').to_pylist()
s=importlib.util.spec_from_file_location('extract','/root/hy4-realworldqa/reference/mcq_extract.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
a=SimpleNamespace(model='hy4-vl',base_url='http://127.0.0.1:8010',max_tokens=16,timeout=180)
for name in ['decode_only','padded_eager']:
    mode(name,'moecapture-'+name)
    save('moecapture-'+name+'-hooks.json',rpc('stage4_exec',str(r/'install_moe_hook.py'),{'label':name}))
    out=evaluate(119,rows[119],a,e.extract_mcq_answer)
    save('moecapture-'+name+'.json',out)
    print(name,out['prediction'],flush=True)
