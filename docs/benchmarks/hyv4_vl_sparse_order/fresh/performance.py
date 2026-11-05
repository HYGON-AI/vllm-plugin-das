import json,requests,subprocess
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh');base='http://127.0.0.1:8010'
def save(n,x):(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rpc(method,*args):
 x=requests.post(base+'/collective_rpc',json={'method':method,'args':list(args),'timeout':120},timeout=150);x.raise_for_status();return x.json()
rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':False})
for label,name in [('perf-full-piecewise','full_piecewise'),('perf-eager','eager')]:
 save(label+'-order.json',rpc('stage4_exec',str(r/'set_order.py'),{'enabled':True}))
 save(label+'-mode.json',rpc('stage4_set_mode',name))
 for path in ['reset_encoder_cache','reset_mm_cache']:
  x=requests.post(base+'/'+path,timeout=60);x.raise_for_status()
 print('START',label,flush=True)
 with (r/(label+'.log')).open('w') as log:
  subprocess.run(['python','tools/hyv4_vl_perf.py','--data-dir','/root/hy4-realworldqa/data/data','--output-dir',str(r/label)],cwd='/tmp/vllm-plugin-das-pr',stdout=log,stderr=subprocess.STDOUT,check=True)
 save(label+'-dispatch.json',rpc('stage4_stats'))
 print('DONE',label,flush=True)
print('DONE performance',flush=True)
