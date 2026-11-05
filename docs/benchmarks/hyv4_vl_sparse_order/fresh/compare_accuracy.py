import json
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
p=r/'accuracy/results.jsonl'
if not p.exists():raise SystemExit('accuracy has not started')
new=[json.loads(s) for s in p.read_text().splitlines()]
ref={v['index']:v for v in [json.loads(s) for s in Path('/root/hy4-perf-stage4/accuracy-decode_only/results.jsonl').read_text().splitlines()]}
changes=[]
for v in new:
 old=ref[v['index']]
 assert (v['question'],v['answer'],v['image_sha256'])==(old['question'],old['answer'],old['image_sha256'])
 if v['prediction']!=old['prediction']:changes.append({'index':v['index'],'before':old['prediction'],'after':v['prediction'],'before_correct':old['correct'],'after_correct':v['correct']})
report={'completed':len(new),'correct':sum(v['correct'] for v in new),'strict_correct':sum(v['strict_correct'] for v in new),'errors':sum(v.get('status')!=200 for v in new),'empty':sum(not v.get('prediction') for v in new),'truncated':sum(v.get('finish_reason')=='length' for v in new),'changed_raw_answers':len(changes),'changes':changes,'reference':'stage4 accuracy-decode_only, 561/765; original question/scorer unchanged'}
(r/'accuracy-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
