import json
import subprocess
import sys
sys.path.insert(0,'/root/hy4-perf-stage4')
from repro import rpc,mode,save,r

def run(label,cmd):
    print('START',label,flush=True)
    rpc('stage4_exec',str(r/'reset_actual.py'))
    with (r/(label+'.log')).open('w') as out:
        subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,cwd='/tmp/vllm-plugin-das-pr')
    save(label+'-dispatcher-attempts.json',rpc('stage4_stats'))
    save(label+'-actual-dispatch.json',rpc('stage4_exec',str(r/'read_actual.py')))
    print('DONE',label,flush=True)
for name,label in [('full_piecewise','accuracy-guard'),('decode_only','accuracy-decode_only')]:
    mode(name,label)
    run(label,['python','tools/hyv4_realworldqa_eval.py','--data-dir','/root/hy4-realworldqa/data/data','--output-dir',str(r/label),'--mcq-extractor','/root/hy4-realworldqa/reference/mcq_extract.py','--max-pixels','1048576','--concurrency','1'])
for name,label in [('full_piecewise','e2e-guard'),('decode_only','e2e-decode_only')]:
    mode(name,label)
    run(label,['python','tools/hyv4_vl_perf.py','--data-dir','/root/hy4-realworldqa/data/data','--output-dir',str(r/label)])
print('DONE ALL',flush=True)
