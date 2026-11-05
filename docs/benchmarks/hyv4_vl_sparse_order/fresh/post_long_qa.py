import subprocess,time
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
for _ in range(1800):
 p=r/'long_qa.py.log'
 if p.exists() and 'DONE long QA' in p.read_text():break
 time.sleep(2)
else:raise RuntimeError('long QA did not finish')
with (r/'performance.py.log').open('w') as log:
 subprocess.run(['python',str(r/'performance.py')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd='/tmp/vllm-plugin-das-pr')
print('DONE performance',flush=True)
