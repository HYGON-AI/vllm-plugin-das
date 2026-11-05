from pathlib import Path
p=Path('/root/hy4-pr-ready-validation/run.py');s=p.read_text();ns={};exec(compile(s[:s.index('for i in range(900):')],str(p),'exec'),ns);ns['r']=Path('/root/hy4-precision-fix')
mode,rpc,save,request,body,r,base=[ns[k] for k in ['mode','rpc','save','request','body','r','base']]
mode('eager');save('source-install.json',rpc('stage4_exec',str(r/'install_source.py')))
results=[]
for rep in range(2):
 mode('eager')
 for i,j in [(112,507),(422,204),(287,719),(478,618)]:
  rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':i==478})
  out=request(body(i,j),base);out.update(repetition=rep,index=i,second_index=j)
  if i==478:save(f'source-{rep}-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/f'source-{rep}.pt')}))
  results.append(out);save('source-eager-answers.json',results);print(rep,i,repr(out['answer']),flush=True)
print('DONE',flush=True)
