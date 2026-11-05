import subprocess,time,json,requests
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh');base='http://127.0.0.1:8010'
for _ in range(2100):
 p=r/'performance.py.log'
 if p.exists() and 'DONE performance' in p.read_text():break
 time.sleep(2)
else:raise RuntimeError('performance did not finish')
x=requests.post(base+'/collective_rpc',json={'method':'stage4_set_mode','args':['full_piecewise'],'timeout':120},timeout=150);x.raise_for_status()
(r/'final-mode.json').write_text(json.dumps(x.json(),indent=2)+'\n')
with (r/'smoke.log').open('w') as log:
 subprocess.run(['python','tools/hyv4_vl_image_smoke.py','--output-dir',str(r/'smoke'),'--photo','/root/hy4-vl-validation/cherry_blossom.jpg'],cwd='/tmp/vllm-plugin-das-pr',stdout=log,stderr=subprocess.STDOUT,check=True)
print('DONE smoke',flush=True)
