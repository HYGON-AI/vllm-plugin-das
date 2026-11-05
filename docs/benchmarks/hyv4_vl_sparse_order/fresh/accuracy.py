import requests,json,time,subprocess
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh');base='http://127.0.0.1:8010'
def save(name,x):(r/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rpc(method,*args):
 x=requests.post(base+'/collective_rpc',json={'method':method,'args':list(args),'timeout':120},timeout=150);x.raise_for_status();return x.json()
for i in range(450):
 try:
  if requests.get(base+'/health',timeout=2).status_code==200:break
 except requests.RequestException:pass
 time.sleep(2)
else:raise RuntimeError('server not ready')
save('mode.json',rpc('stage4_set_mode','full_piecewise'))
save('source-flags.json',rpc('stage4_exec',str(r/'install_counters.py')))
for endpoint in ['reset_encoder_cache','reset_mm_cache']:
 x=requests.post(base+'/'+endpoint,timeout=60);x.raise_for_status()
print('START full accuracy',flush=True)
with (r/'accuracy.log').open('w') as log:
 subprocess.run(['python','tools/hyv4_realworldqa_eval.py','--data-dir','/root/hy4-realworldqa/data/data','--output-dir',str(r/'accuracy'),'--mcq-extractor','/root/hy4-realworldqa/reference/mcq_extract.py','--max-pixels','1048576','--concurrency','1'],cwd='/tmp/vllm-plugin-das-pr',stdout=log,stderr=subprocess.STDOUT,check=True)
save('accuracy-actual-dispatch.json',rpc('stage4_exec',str(r/'read_counters.py')))
save('accuracy-dispatcher-attempts.json',rpc('stage4_stats'))
print('DONE full accuracy',flush=True)
