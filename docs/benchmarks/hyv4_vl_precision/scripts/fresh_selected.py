import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pyarrow.parquet as pq
sys.path.insert(0,'/tmp/vllm-plugin-das-pr')
from tools.hyv4_realworldqa_eval import evaluate
r=Path('/root/hy4-perf-stage4')
rows=[]
for path in sorted(Path('/root/hy4-realworldqa/data/data').glob('*.parquet')):rows.extend(pq.read_table(path).to_pylist())
s=importlib.util.spec_from_file_location('extract','/root/hy4-realworldqa/reference/mcq_extract.py');ex=importlib.util.module_from_spec(s);s.loader.exec_module(ex)
a=SimpleNamespace(model='hy4-vl',base_url='http://127.0.0.1:8010',max_tokens=16,timeout=180)
baseline={x['index']:x for x in map(json.loads,open('/root/hy4-perf-stage3/accuracy-decode_only/results.jsonl'))}
out=[]
for idx in [0,45,94,119,126,143,155,170,214,305,313,374,401,506,598]:
    x=evaluate(idx,rows[idx],a,ex.extract_mcq_answer);out.append(x)
    (r/'fresh-selected.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(idx,repr(x['prediction']),flush=True)
    assert x.get('status')==200 and x['prediction']==baseline[idx]['prediction']
print('DONE',flush=True)
