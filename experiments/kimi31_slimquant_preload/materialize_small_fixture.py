"""Create a four-layer, two-expert diagnostic fixture with verified data.

The six gate tensors are truncated to two rows; all other selected tensors
are copied unchanged. This is a diagnostic model, not the full checkpoint.
"""
import hashlib,json,os,re,struct
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Materialize a diagnostic four-layer two-expert fixture.')
parser.add_argument('--source-model', type=Path, required=True,
                    help='source checkpoint directory (config and safetensors index)')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'fixture_4layer_2expert_v1')
args = parser.parse_args()
root = args.source_model.resolve()
if not root.is_dir() or not (root/'model.safetensors.index.json').is_file():
    parser.error('--source-model must contain model.safetensors.index.json')
base=Path(__file__).parent
out=args.output_dir.resolve()
tmp=out.with_name(out.name+'.inprogress')
if out.exists() or tmp.exists(): raise FileExistsError('fixture or in-progress directory exists')
if not (root/'config.json').is_file(): parser.error('--source-model must contain config.json')
raw_config=(root/'config.json').read_bytes(); raw_index=(root/'model.safetensors.index.json').read_bytes()
out.parent.mkdir(parents=True, exist_ok=True)
tmp.mkdir()
config=json.loads(raw_config); original_index=json.loads(raw_index)['weight_map']
text=config['text_config']; text['num_hidden_layers']=4; text['num_experts']=2
text['num_experts_per_token']=2; text['min_moe_intermediate_per_partition']=192
config['architectures']=['HcuKimi31SlimQuantLoadOnly']
(tmp/'config.json').write_text(json.dumps(config,indent=2)+'\n')
selected={}
for name,shard in original_index.items():
    layer=re.search(r'\.layers\.(\d+)\.',name)
    if name.startswith('language_model.') and layer and int(layer.group(1))>=4: continue
    expert=re.search(r'\.block_sparse_moe\.experts\.(\d+)\.',name)
    if expert and int(expert.group(1))>=2: continue
    selected[name]=shard
manifest={'scope':'diagnostic four-layer two-expert fixture; gate tensors cropped to first two experts',
  'source':str(root),'source_config_sha256':hashlib.sha256(raw_config).hexdigest(),
  'source_index_sha256':hashlib.sha256(raw_index).hexdigest(),'tensors':{},'shards':{}}
weight_map={}
for number,shard in enumerate(sorted(set(selected.values())),1):
    names=[name for name in selected if selected[name]==shard]
    src=root/shard
    with src.open('rb') as stream:
        n=struct.unpack('<Q',stream.read(8))[0]; header=json.loads(stream.read(n)); start_data=8+n
        names.sort(key=lambda x:header[x]['data_offsets'][0])
        new_header={}; offset=0
        for name in names:
            info=header[name]; start,end=info['data_offsets']; shape=list(info['shape']); transformed=False
            if name.endswith('.block_sparse_moe.gate.weight'):
                assert shape==[896,7168] and info['dtype']=='BF16'; shape[0]=2; end=start+2*7168*2; transformed=True
            elif name.endswith('.block_sparse_moe.gate.e_score_correction_bias'):
                assert shape==[896] and info['dtype']=='F32'; shape[0]=2; end=start+2*4; transformed=True
            length=end-start
            new_header[name]={'dtype':info['dtype'],'shape':shape,'data_offsets':[offset,offset+length]}
            manifest['tensors'][name]={'source_shard':shard,'source_offsets':[start,end], 'cropped':transformed}
            offset+=length
        new_header['__metadata__']={'format':'pt'}
        raw=json.dumps(new_header,separators=(',',':')).encode(); raw+=b' '*(-len(raw)%8)
        filename='fixture-%05d-of-%05d.safetensors' % (number,len(set(selected.values())))
        with (tmp/filename).open('xb') as output:
            output.write(struct.pack('<Q',len(raw))); output.write(raw)
            for name in names:
                start,end=manifest['tensors'][name]['source_offsets']; stream.seek(start_data+start)
                digest=hashlib.sha256(); remaining=end-start
                while remaining:
                    block=stream.read(min(8<<20,remaining))
                    if not block: raise ValueError('short source: '+name)
                    output.write(block); digest.update(block); remaining-=len(block)
                manifest['tensors'][name]['sha256']=digest.hexdigest(); weight_map[name]=filename
            output.flush(); os.fsync(output.fileno())
        manifest['shards'][filename]={'source':shard,'tensor_count':len(names),'data_bytes':offset}
    # Re-read each target tensor from disk; verify content against the source stream.
    with (tmp/filename).open('rb') as check:
        hdrlen=struct.unpack('<Q',check.read(8))[0]; hdr=json.loads(check.read(hdrlen)); data_start=8+hdrlen
        for name in names:
            start,end=hdr[name]['data_offsets'];check.seek(data_start+start); digest=hashlib.sha256(); remaining=end-start
            while remaining:
                block=check.read(min(8<<20,remaining))
                if not block: raise ValueError('short fixture: '+name)
                digest.update(block); remaining-=len(block)
            if digest.hexdigest()!=manifest['tensors'][name]['sha256']: raise ValueError('mismatch: '+name)
    print('verified',filename,len(names),offset,flush=True)
(tmp/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(x['data_bytes'] for x in manifest['shards'].values())},'weight_map':weight_map},indent=2)+'\n')
for asset in root.iterdir():
    if asset.name in ('config.json','model.safetensors.index.json') or asset.name.endswith('.safetensors'): continue
    if asset.is_file(): (tmp/asset.name).symlink_to(asset)
(tmp/'fixture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if out.exists(): raise FileExistsError(out)
tmp.rename(out)
print('fixture',out,'tensors',len(weight_map),flush=True)
