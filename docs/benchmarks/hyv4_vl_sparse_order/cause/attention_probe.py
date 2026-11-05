from pathlib import Path
p=Path('/root/hy4-pr-ready-validation/run.py');s=p.read_text();ns={};exec(compile(s[:s.index('for i in range(900):')],str(p),'exec'),ns);ns['r']=Path('/root/hy4-precision-fix')
mode,rpc,save,request,body,r,base=[ns[k] for k in ['mode','rpc','save','request','body','r','base']]
for selector in ['original','canonical']:
 mode('eager');rpc('stage4_exec',str(r/'selector_install.py'),{'selector':selector})
 rpc('stage4_exec',str(r/'capture_attention.py'))
 rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':True})
 out=request(body(478,618),base);save(f'attention-{selector}-answer.json',out)
 save(f'attention-{selector}-capture.json',rpc('stage4_exec',str(r/'save_attention.py'),{'path':str(r/f'attention-{selector}.pt')}))
 print(selector,repr(out['answer']),flush=True)
