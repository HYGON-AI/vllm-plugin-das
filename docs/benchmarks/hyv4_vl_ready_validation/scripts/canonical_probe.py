import runpy
# Load request helpers without executing the previous validation sequence.
from pathlib import Path
p=Path('/root/hy4-pr-ready-validation/run.py')
s=p.read_text();s=s[:s.index('for i in range(900):')]
ns={};exec(compile(s,str(p),'exec'),ns)
mode,rpc,save,request,body,r,base=[ns[k] for k in ['mode','rpc','save','request','body','r','base']]
results=[]
for selector in ['canonical']:
    for repetition in range(3):
        mode('eager')
        save(f'topk-{selector}-{repetition}-install.json',rpc('stage4_exec',str(r/'topk_probe_install.py'),{'selector':selector}))
        rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':True})
        out=request(body(478,618),base)
        out.update(selector=selector,repetition=repetition)
        label=f'topk-{selector}-{repetition}'
        save(label+'-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/(label+'.pt'))}))
        save(label+'-selection.json',rpc('stage4_exec',str(r/'topk_probe_save.py'),{'path':str(r/(label+'-selection.pt'))}))
        results.append(out);save('topk-canonical-ablation.json',results)
        print(label,repr(out['answer']),flush=True)
print('DONE',flush=True)
