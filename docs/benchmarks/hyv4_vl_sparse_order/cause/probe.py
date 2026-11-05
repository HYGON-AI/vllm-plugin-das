from pathlib import Path
p=Path('/root/hy4-pr-ready-validation/run.py');s=p.read_text();ns={};exec(compile(s[:s.index('for i in range(900):')],str(p),'exec'),ns)
ns['r']=Path('/root/hy4-precision-fix')
mode,rpc,save,request,body,r,base=[ns[k] for k in ['mode','rpc','save','request','body','r','base']]
import time,requests
for i in range(450):
 try:
  if requests.get(base+'/health',timeout=2).status_code==200:break
 except requests.RequestException:pass
 time.sleep(2)
else:raise RuntimeError('server not ready')
mode('eager');save('trace-installed.json',rpc('stage4_exec',str(r/'trace_install.py')))
results=[]
for selector in ['original','canonical','stable']:
 for rep in range(2):
  mode('eager')
  save(f'{selector}-{rep}-install.json',rpc('stage4_exec',str(r/'selector_install.py'),{'selector':selector}))
  for i,j in [(112,507),(422,204),(287,719),(478,618)]:
   trace=i==478
   rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':trace})
   out=request(body(i,j),base);out.update(selector=selector,repetition=rep,index=i,second_index=j)
   label=f'{selector}-{rep}-{i}-{j}'
   if trace:
    save(label+'-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/(label+'.pt'))}))
    save(label+'-topk.json',rpc('stage4_exec',str(r/'save_selector.py'),{'path':str(r/(label+'-topk'))}))
   results.append(out);save('selector-ablation.json',results)
   print(label,repr(out['answer']),flush=True)
print('DONE',flush=True)
