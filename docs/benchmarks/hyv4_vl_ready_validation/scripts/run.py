import base64
import io
import json
from pathlib import Path
import sys
import time
import pyarrow.parquet as pq
import requests
from PIL import Image
sys.path.insert(0,'/tmp/vllm-plugin-das-pr')
from tools.hyv4_vl_perf import request
r=Path('/root/hy4-pr-ready-validation')
base='http://127.0.0.1:8010'
def save(name,value):(r/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def rpc(method,*args):
    res=requests.post(base+'/collective_rpc',json={'method':method,'args':list(args),'timeout':120},timeout=150)
    res.raise_for_status();return res.json()
def mode(name):
    rpc('stage4_set_mode',name)
    for endpoint in ['reset_encoder_cache','reset_mm_cache']:
        res=requests.post(base+'/'+endpoint,timeout=60);res.raise_for_status()
rows=[]
for p in sorted(Path('/root/hy4-realworldqa/data/data').glob('*.parquet')):rows.extend(pq.read_table(p).to_pylist())
urls={}
for i in [112,507,422,204,287,719,478,618]:
    im=Image.open(io.BytesIO(rows[i]['image']['bytes'])).convert('RGB');buffer=io.BytesIO();im.save(buffer,format='PNG')
    urls[i]='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
def body(i,j,limit=32):
    return {'model':'hy4-vl','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':urls[i]}},{'type':'image_url','image_url':{'url':urls[j]}},{'type':'text','text':'Describe both images in detail, in their original order.'}]}], 'temperature':0,'seed':0,'max_tokens':limit,'min_tokens':limit,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'reasoning_effort':'no_think'}}
for i in range(900):
    try:
        if requests.get(base+'/health',timeout=2).status_code==200:break
    except requests.RequestException:pass
    if i%15==0:print('WAIT',i*2,flush=True)
    time.sleep(2)
else:raise RuntimeError('readiness timeout')
mode('full_piecewise');save('trace-installed.json',rpc('stage4_exec',str(r/'trace_install.py')))
results=[]
for repetition in range(2):
    for name in ['full_piecewise','padded_eager','decode_only','eager']:
        mode(name)
        for i,j in [(112,507),(422,204),(287,719),(478,618)]:
            label=f'{repetition}-{name}-{i}-{j}'
            trace=i==478
            rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':trace})
            result=request(body(i,j),base)
            result.update(repetition=repetition,mode=name,index=i,second_index=j)
            if trace:save(label+'-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/(label+'.pt'))}))
            results.append(result);save('two-image-repeats.json',results)
            print(label,repr(result['answer']),flush=True)
        save(f'{repetition}-{name}-dispatch.json',rpc('stage4_stats'))
print('DONE',flush=True)
