import subprocess,time
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
for _ in range(1500):
 p=r/'captions.py.log'
 if p.exists() and 'DONE captions' in p.read_text():break
 time.sleep(2)
else:raise RuntimeError('captions did not finish')
with (r/'long_qa.py.log').open('w') as log:
 subprocess.run(['python',str(r/'long_qa.py')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd='/tmp/vllm-plugin-das-pr')
print('DONE long QA',flush=True)
