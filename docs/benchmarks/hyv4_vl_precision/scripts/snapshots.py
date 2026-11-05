import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pyarrow.parquet as pq
sys.path.insert(0,'/root/hy4-perf-stage4')
from repro import rpc,mode,save,r
sys.path.insert(0,'/tmp/vllm-plugin-das-pr')
from tools.hyv4_realworldqa_eval import evaluate
rows=[]
for p in sorted(Path('/root/hy4-realworldqa/data/data').glob('*.parquet')):
    rows.extend(pq.read_table(p).to_pylist())
spec=importlib.util.spec_from_file_location('extract','/root/hy4-realworldqa/reference/mcq_extract.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
args=SimpleNamespace(model='hy4-vl',base_url='http://127.0.0.1:8010',max_tokens=16,timeout=180)
mods=rpc('stage4_modules','model.layers.0')['results'][0]
names=[n for n,cls in mods]
names+=['model.layers.'+str(i) for i in range(1,78)]
names+=['model.hc_head','model.norm']
for idx in [119,94]:
    for it,name in enumerate(['decode_only','padded_eager','decode_only','padded_eager']):
        label=f'snapshot-{idx}-{it}-{name}'
        mode(name,label)
        save(label+'-hooks.json',rpc('stage4_snapshot',str(r/(label+'.pt')),names))
        result=evaluate(idx,rows[idx],args,ex.extract_mcq_answer)
        save(label+'.json',result)
        save(label+'-flush.json',rpc('stage4_flush'))
        print(label,repr(result.get('prediction')),flush=True)
        rpc('stage4_snapshot','',[])
print('DONE',flush=True)
