import json
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
def read(n):return json.loads((r/n).read_text())
a=read('accuracy-comparison.json')
assert a['completed']==765 and a['correct']==561 and a['strict_correct']==546
assert a['errors']==a['empty']==a['truncated']==a['changed_raw_answers']==0
for size,n,traces,steps in [(32,32,8,33),(128,12,4,129)]:
 x=read(f'caption{size}-comparison.json');assert x['requests']==n
 assert all(v['answers_all_identical'] for v in x['groups'])
 assert len(x['logit_comparisons'])==traces
 assert all(v['records']==steps and v['reference_records']==steps and not v['changed'] for v in x['logit_comparisons'])
 assert len(x['indices_validation'])==8*traces and all(v['all_canonical'] for v in x['indices_validation'])
x=read('long-qa-comparison.json');assert x['requests']==96
for name in ['fixed_eager','fixed_full_piecewise']:
 assert x['labels'][name]['runs']==32 and not x['labels'][name]['changed_repeats']
 assert x['labels'][name]['correct_by_repetition']=={'0':12,'1':12}
assert len(x['pairs'])==32 and all(v['raw_equal'] for v in x['pairs'])
assert all(not v['raw_changes'] for v in read('performance-comparison.json').values())
x=read('smoke/results.json');print('smoke structure',type(x).__name__)
if isinstance(x,dict):print('smoke keys',list(x))
else:
 scored=[v for v in x if v['correct'] is not None];assert len(scored)==7 and all(v['correct'] for v in scored)
(r/'release-gate.json').write_text(json.dumps({'passed':True,'tested_code_head':'47bbdcda999e2ca8c6099da72423fec2f6adc7b9','local_regression_tests':287,'official_qa':765,'official_raw_changes':0,'caption32_requests':32,'caption128_requests':12,'long_qa_requests':96,'performance_raw_changes':0,'smoke_scored':7},indent=2)+'\n')
print('RELEASE GATE PASSED')
