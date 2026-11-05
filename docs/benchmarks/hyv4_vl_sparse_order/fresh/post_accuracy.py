import time,subprocess
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
for _ in range(1200):
 if (r/'accuracy-actual-dispatch.json').exists():break
 time.sleep(2)
else:raise RuntimeError('full accuracy did not finish')
for name in ['compare_accuracy.py','captions.py']:
 print('START',name,flush=True)
 with (r/(name+'.log')).open('w') as log:subprocess.run(['python',str(r/name)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd='/tmp/vllm-plugin-das-pr')
 print('DONE',name,flush=True)
print('DONE post-accuracy',flush=True)
