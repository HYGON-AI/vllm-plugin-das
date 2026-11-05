from pathlib import Path
p=Path('/root/hy4-pr-ready-validation/run.py');s=p.read_text();ns={};exec(compile(s[:s.index('for i in range(900):')],str(p),'exec'),ns);ns['r']=Path('/root/hy4-precision-fix/fresh')
mode,rpc,save,request,body,r,base=[ns[k] for k in ['mode','rpc','save','request','body','r','base']]
mode('eager');save('trace-install.json',rpc('stage4_exec',str(r/'trace_install.py')))
results=[]
for repetition in range(2):
 for name in ['full_piecewise','padded_eager','decode_only','eager']:
  mode(name)
  for i,j in [(112,507),(422,204),(287,719),(478,618)]:
   trace=i==478
   rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':trace})
   out=request(body(i,j),base);out.update(repetition=repetition,mode=name,index=i,second_index=j)
   label=f'caption32-{repetition}-{name}-{i}-{j}'
   if trace:save(label+'-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/label)}))
   results.append(out);save('caption32-results.json',results);print(label,repr(out['answer']),flush=True)
  save(f'caption32-{repetition}-{name}-dispatch.json',rpc('stage4_stats'))
# Extend generation well beyond the first 2049-token boundary.
long=[]
for repetition in range(2):
 for name in ['full_piecewise','eager']:
  mode(name)
  for i,j in [(112,507),(287,719),(478,618)]:
   rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':i==478})
   out=request(body(i,j,128),base);out.update(repetition=repetition,mode=name,index=i,second_index=j)
   label=f'caption128-{repetition}-{name}-{i}-{j}'
   if i==478:save(label+'-trace.json',rpc('stage4_exec',str(r/'trace_save.py'),{'path':str(r/label)}))
   long.append(out);save('caption128-results.json',long);print(label,'done',flush=True)
rpc('stage4_exec',str(r/'trace_reset.py'),{'enabled':False})
print('DONE captions',flush=True)
