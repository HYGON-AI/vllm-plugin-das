"""Plan a small four-layer/two-expert fixture; metadata only."""
import collections,json,re,struct
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Plan a diagnostic fixture from checkpoint metadata.')
parser.add_argument('--source-model', type=Path, required=True,
                    help='source checkpoint directory (config and safetensors index)')
parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
args = parser.parse_args()
root = args.source_model.resolve()
if not root.is_dir() or not (root/'model.safetensors.index.json').is_file():
    parser.error('--source-model must contain model.safetensors.index.json')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
selected={}
for name,shard in index.items():
    m=re.search(r'\.layers\.(\d+)\.',name)
    if name.startswith('language_model.') and m and int(m.group(1))>=4: continue
    e=re.search(r'\.block_sparse_moe\.experts\.(\d+)\.',name)
    if e and int(e.group(1))>=2: continue
    selected[name]=shard
sizes=collections.Counter();counts=collections.Counter();transformed=[]
for shard in set(selected.values()):
    with (root/shard).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(size))
    for name,filename in selected.items():
        if filename != shard: continue
        entry=header[name]; start,end=entry['data_offsets']; data_bytes=end-start
        if name.endswith('.block_sparse_moe.gate.weight'):
            assert entry['shape']==[896,7168] and entry['dtype']=='BF16'
            data_bytes=2*7168*2; transformed.append(name)
        elif name.endswith('.block_sparse_moe.gate.e_score_correction_bias'):
            assert entry['shape']==[896] and entry['dtype']=='F32'
            data_bytes=2*4; transformed.append(name)
        sizes[shard]+=data_bytes;counts[shard]+=1
out={'scope':'fixture metadata plan only; no checkpoint payload read', 'layers':4,'experts':2,
     'selected_tensors':len(selected),'selected_bytes':sum(sizes.values()),
     'source_shards':len(sizes),'largest_output_shard_bytes':max(sizes.values()),
     'per_shard':[{'shard':k,'bytes':sizes[k],'tensors':counts[k]} for k in sorted(sizes)],
     'transformed':transformed}
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
