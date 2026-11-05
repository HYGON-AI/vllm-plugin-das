import base64,io,json,random,sys,importlib.util
from pathlib import Path
import requests,pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0,'/tmp/vllm-plugin-das-pr')
from tools.hyv4_vl_perf import request
from tools.hyv4_realworldqa_eval import score
r=Path('/root/hy4-precision-fix/fresh');base='http://127.0.0.1:8010'
def save(n,x):(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rpc(method,*args):
 x=requests.post(base+'/collective_rpc',json={'method':method,'args':list(args),'timeout':120},timeout=150);x.raise_for_status();return x.json()
def mode(name):
 rpc('stage4_set_mode',name)
 for path in ['reset_encoder_cache','reset_mm_cache']:
  x=requests.post(base+'/'+path,timeout=60);x.raise_for_status()
rows=[v for p in sorted(Path('/root/hy4-realworldqa/data/data').glob('*.parquet')) for v in pq.read_table(p).to_pylist()]
indices=random.Random(338).sample(range(len(rows)),16)
urls={}
for idx in indices:
 im=Image.open(io.BytesIO(rows[idx]['image']['bytes'])).convert('RGB');b=io.BytesIO();im.save(b,format='PNG');urls[idx]='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
spec=importlib.util.spec_from_file_location('mcq','/root/hy4-realworldqa/reference/mcq_extract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=[];preparation=[]
for pos,idx in enumerate(indices):
 second=indices[(pos+4)%len(indices)];padding=0;lo=0;hi=2048
 def body(padding):
  text='Ignore the following filler context: '+('0 '*padding)+'\nEnd filler context. Answer the question about the FIRST image only.\n'+rows[idx]['question'].strip()
  return {'model':'hy4-vl','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':urls[idx]}},{'type':'image_url','image_url':{'url':urls[second]}},{'type':'text','text':text}]}],'temperature':0,'top_p':1.0,'max_tokens':16,'seed':0,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'reasoning_effort':'no_think'}}
 for _ in range(14):
  padding=(lo+hi)//2
  payload=body(padding)
  tok=requests.post(base+'/tokenize',json={k:payload[k] for k in ['model','messages','chat_template_kwargs']},timeout=120);tok.raise_for_status();n=tok.json()['count']
  if 3072<=n<=3076:break
  if n<3072:lo=padding+1
  else:hi=padding-1
 else:raise RuntimeError(('cannot reach target context',idx,n,padding))
 cases.append((idx,second,payload));preparation.append({'index':idx,'second_index':second,'prompt_tokens':n,'zero_words':padding,'question':rows[idx]['question'],'reference':rows[idx]['answer']})
save('long-qa-manifest.json',{'sample_seed':338,'indices':indices,'prompt_is_modified':True,'purpose':'two real images plus ignored text filler, first-image QA at 3072 tokens; not the official dataset score','cases':preparation})
print('PREPARED',len(cases),flush=True)
results=[]
for label,mode_name,enabled in [('native_eager','eager',False),('fixed_eager','eager',True),('fixed_full_piecewise','full_piecewise',True)]:
 save(label+'-order.json',rpc('stage4_exec',str(r/'set_order.py'),{'enabled':enabled}))
 for rep in range(2):
  mode(mode_name)
  for idx,second,payload in cases:
   out=request(payload,base);norm,ok=score(rows[idx]['answer'],out['answer'],m.extract_mcq_answer)
   out.update(index=idx,second_index=second,label=label,repetition=rep,reference=rows[idx]['answer'],normalized_prediction=norm,correct=ok)
   results.append(out);save('long-qa-results.json',results);print(label,rep,idx,repr(out['answer']),ok,flush=True)
 print(label,'DONE',flush=True)
rpc('stage4_exec',str(r/'set_order.py'),{'enabled':True})
print('DONE long QA',flush=True)
