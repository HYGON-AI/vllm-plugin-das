"""Check real checkpoint expert TP16 slicing; no vLLM or GPU allocation."""
import os
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from hcu_vllm_das.quantization.slimquant_shard import shard_expert

root=Path(os.environ["KIMI_K31_CHECKPOINT"])
prefix='language_model.model.layers.1.block_sparse_moe.experts.0.'
keys=[prefix+part+suffix for part in ('w1','w2','w3') for suffix in ('.weight','.weight_scale')]
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
values={}
for shard in set(index[k] for k in keys):
    with safe_open(str(root/shard),framework='pt',device='cpu') as f:
        for name in keys:
            if index[name] == shard:
                values[name] = f.get_tensor(name).clone()
w1,w3,w2=(values[prefix+x+'.weight'] for x in ('w1','w3','w2'))
s1,s3,s2=(values[prefix+x+'.weight_scale'] for x in ('w1','w3','w2'))
assert w1.shape == (3072,1792) and w2.shape == (3584,1536)
rank_hashes=[]
for rank in range(16):
    a,b,sa,sb=shard_expert(w1,w3,w2,s1,s3,s2,rank=rank,tp_size=16,padded_intermediate=256)
    start=rank*192
    assert torch.equal(a[:192],w1[start:start+192])
    assert torch.equal(a[256:448],w3[start:start+192])
    assert torch.equal(b[:,:96],w2[:,rank*96:(rank+1)*96])
    assert torch.equal(sa[:192],s1[start:start+192]) and torch.equal(sa[256:448],s3[start:start+192])
    assert torch.equal(sb,s2)
    assert not torch.count_nonzero(a[192:256]) and not torch.count_nonzero(a[448:]) and not torch.count_nonzero(b[:,96:])
    assert torch.all(sa[192:256] == 1) and torch.all(sa[448:] == 1)
    rank_hashes.append(hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()+sa.numpy().tobytes()+sb.numpy().tobytes()).hexdigest())
out={'checkpoint':str(root),'scope':'one expert, local CPU slices only; no vLLM, GPU, IPC or numerical inference',
     'rank_count':16,'logical_intermediate_per_rank':192,'padded_intermediate_per_rank':256,
     'checkpoint_tensor_shards':{key:index[key] for key in keys},'rank_sha256':rank_hashes,'passed':True}
Path(__file__).with_name('one_expert_tp16.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
