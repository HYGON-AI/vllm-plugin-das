# SPDX-License-Identifier: Apache-2.0
"""Run the supplied C64/64K->1 benchmark from an installed Step5 environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client',type=Path,required=True)
    p.add_argument('--capacities',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--label',default='fresh-release')
    a=p.parse_args()
    import re
    if not re.fullmatch('[a-z0-9-]+',a.label):raise ValueError('Use a unique lowercase trial label')
    cfg=json.load(urllib.request.urlopen('http://127.0.0.1:8018/server_info?config_format=json',timeout=20))['vllm_config']
    pc,sp,sc=cfg['parallel_config'],cfg['speculative_config'],cfg['scheduler_config']
    if (pc['data_parallel_size'],pc['tensor_parallel_size'])!=(16,1) or not pc['enable_expert_parallel'] or not pc['enable_eplb']:
        raise ValueError('Expected DP16/TP1/EP16 and fixed E23')
    if pc['all2all_backend']!='deepep_high_throughput' or pc['eplb_config']['num_redundant_experts']!=16:
        raise ValueError('Unexpected communication or expert configuration')
    if sc['max_num_batched_tokens']!=8192 or cfg['model_config']['max_model_len']!=70000:
        raise ValueError('Unexpected token budget or context limit')
    if sp['method']!='mtp' or sp['num_speculative_tokens']!=3 or sp['rejection_sample_method']!='synthetic' or sp['synthetic_acceptance_rates']!=[.838992,.598322,.362686]:
        raise ValueError('The fixed synthetic MTP3 configuration is required')
    if cfg['cache_config']['enable_prefix_caching'] or cfg['cache_config']['cache_dtype']!='fp8_e4m3' or cfg['cache_config']['gpu_memory_utilization']!=.875:
        raise ValueError('Unexpected KV cache configuration')
    caps=json.loads(a.capacities.read_text())
    if set(map(int,caps))!=set(range(16)) or min(caps.values())<=0:raise ValueError('Need all16 current initialized capacities')
    state=a.state.resolve();run=state/'client-runs'/a.label;run.mkdir(parents=True,exist_ok=False)
    files=['run_case.py','benchmark_pd_tps.py','benchmark_hcu_metadata.py','flops_calculator.py']
    hashes={}
    for name in files:
        src=a.client/name;hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest();shutil.copy2(src,run/name)
    (run/'startup-capacities.json').write_text(json.dumps(caps))
    (run/'server-config.json').write_text(json.dumps(cfg,indent=2))
    env=dict(os.environ,STEP5_ITL_STATE=str(state),STEP5_STATE_DIR=str(state),STEP5_TRIAL_NAME=a.label,
             STEP5_NATIVE_BUNDLE=str(run),STEP5_BENCH_PREFILL_TOKENS='8192',PYTHONNOUSERSITE='1',
             STEP5_MODEL_DIR=cfg['model_config']['model'])
    env.pop('PYTHONPATH',None)
    for number in [1,3,4,5]:
        env['STEP5_MEASUREMENT_ROUND']=str(number)
        subprocess.run([sys.executable,'-u',str(run/'run_case.py'),'prefill','64'],cwd=str(run),env=env,check=True)
    scores=[];receipts=[]
    for manifest in sorted((state/'trials'/a.label).glob('*/*_manifest.json')):
        m=json.loads(manifest.read_text());report=json.loads(manifest.with_name(manifest.name.replace('_manifest','')).read_text())
        z=next(v for k,v in report.items() if k.startswith('bs64_') and isinstance(v,dict))
        if not m['completed'] or not m['valid_for_comparison'] or m['num_preemptions']!=0 or z['complete_requests']!=64 or z['incomplete_requests']!=0:
            raise ValueError('Invalid measured workload: '+str(manifest))
        if not m['excluded_from_final']:scores.append(z['prefill_tps'])
        receipts.append(dict(manifest=str(manifest),excluded=m['excluded_from_final'],tps=z['prefill_tps']))
    if len(scores)!=3:raise ValueError('Three formal rounds were not completed')
    result=dict(complete=True,warmup_rounds=1,formal_tps=scores,mean_tps=statistics.mean(scores),
                min_tps=min(scores),max_tps=max(scores),all64_complete=True,preemptions=0,
                synthetic_acceptance_rates=sp['synthetic_acceptance_rates'],client_sha256=hashes,rounds=receipts)
    out=state/(a.label+'-RESULT.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
