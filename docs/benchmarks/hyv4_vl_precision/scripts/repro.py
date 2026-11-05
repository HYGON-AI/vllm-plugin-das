import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pyarrow.parquet as pq
import requests
sys.path.insert(0, '/tmp/vllm-plugin-das-pr')
from tools.hyv4_realworldqa_eval import evaluate
r=Path('/root/hy4-perf-stage4')
base='http://127.0.0.1:8010'
def rpc(method,*args):
    response=requests.post(base+'/collective_rpc',json={'method':method,'args':list(args),'timeout':120},timeout=150)
    response.raise_for_status()
    return response.json()
def save(name,data):
    (r/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def mode(name,label):
    save(label+'-mode.json',rpc('stage4_set_mode',name))
    for endpoint in ('reset_encoder_cache','reset_mm_cache'):
        response=requests.post(base+'/'+endpoint,timeout=60)
        response.raise_for_status()

def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['decode_only','padded_eager','full_piecewise','decode_only','padded_eager','full_piecewise'])
    p.add_argument('--indices',nargs='+',type=int,default=[0,45,94,119,126,143,155,170,214,305,313,374,401,506,598])
    p.add_argument('--label',default='ablation')
    a=p.parse_args()
    for i in range(600):
        try:
            if requests.get(base+'/health',timeout=2).status_code==200:break
        except requests.RequestException:pass
        if i%15==0:print('WAIT',i*2,flush=True)
        time.sleep(2)
    else:raise RuntimeError('server unavailable')
    rows=[]
    for path in sorted(Path('/root/hy4-realworldqa/data/data').glob('*.parquet')):
        rows.extend(pq.read_table(path).to_pylist())
    spec=importlib.util.spec_from_file_location('extract','/root/hy4-realworldqa/reference/mcq_extract.py')
    ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
    args=SimpleNamespace(model='hy4-vl',base_url=base,max_tokens=16,timeout=180)
    save('modules.json',rpc('stage4_modules'))
    for it,name in enumerate(a.modes):
        label=f'{a.label}-{it}-{name}'
        mode(name,label)
        results=[]
        for idx in a.indices:
            result=evaluate(idx,rows[idx],args,ex.extract_mcq_answer)
            results.append(result)
            print(label,idx,repr(result.get('prediction')),result.get('correct'),flush=True)
            save(label+'.json',results)
            if result.get('status')!=200:raise RuntimeError(result)
        save(label+'-dispatch.json',rpc('stage4_stats'))
    print('DONE',flush=True)
if __name__=='__main__':main()
