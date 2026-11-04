# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Explicit diagnostic lane only: force native blocks while comparing real histories."""
import json
from pathlib import Path
import torch


def metrics(ref, actual):
    ref, actual = ref.float(), actual.float()
    assert ref.shape == actual.shape, (ref.shape,actual.shape)
    assert torch.isfinite(ref).all() and torch.isfinite(actual).all()
    delta=actual-ref
    return {'relative_l2':float(delta.norm()/ref.norm()),'max_abs':float(delta.abs().max()),
            'mean_abs':float(delta.abs().mean())}


def compare_and_force(sampler,batch,raw,tokens,counts):
    state=sampler.fast_model_state
    runner=state.fast_runner
    for row,req in enumerate(batch.idx_mapping_np):
        entry=sampler.fast_teachers[int(req)]
        step=entry['step'];meta=entry['manifest']['steps'][step]
        reference=torch.load(entry['folder']/meta['file'],map_location='cpu',weights_only=False)
        expected=reference['generated'][0].tolist()
        total=int(sampler.req_states.total_len.gpu[req].cpu())
        actual=sampler.req_states.all_token_ids.gpu[req,:total].cpu().tolist()
        assert actual==expected, ('teacher real history differs',step,len(actual),len(expected))
        assert total==meta['real_length']
        ref=reference['raw_logits'][0]
        value=raw[row*6:(row+1)*6]
        report={'sample_index':entry['manifest']['sample_index'],'step':step,'history_exact':True,
                'real_length':total,'logits':metrics(ref,value),
                'argmax_agreement':float((ref.argmax(-1)==value.argmax(-1)).float().mean()),
                'candidate_tokens':tokens[row,:int(counts[row].cpu())].cpu().tolist(),
                'reference_tokens':meta['output'],'purpose':'teacher_forced_diagnostic','performance_valid':False}
        positions=torch.tensor(reference['kv_positions'],device=tokens.device)
        table=runner.block_tables.block_tables[0].gpu[req]
        block_size=runner.block_tables.kernel_block_sizes[0]
        physical=table[positions//block_size].long()
        slots=positions%block_size
        layer_reports={}
        for ix,layer in enumerate(reference['kv_layers']):
            cache=state.model.language_model.model.layers[layer].self_attn.attn.kv_cache
            assert torch.is_tensor(cache),type(cache)
            fetched=cache[physical,:,slots,:].cpu()
            key,value=fetched.split(128,dim=-1)
            layer_reports[str(layer)]={'key':metrics(reference['kv'][ix,0],key),
                                      'value':metrics(reference['kv'][ix,1],value)}
        report['kv']=layer_reports
        folder=entry['folder'].parent/'comparison'/str(entry['manifest']['sample_index'])
        folder.mkdir(parents=True,exist_ok=True)
        (folder/f'{step:04d}.json').write_text(json.dumps(report,indent=2))
        n=len(meta['output']);tokens[row].zero_()
        tokens[row,:n]=torch.tensor(meta['output'],device=tokens.device,dtype=tokens.dtype)
        counts[row]=n;entry['step']+=1
    return tokens,counts
