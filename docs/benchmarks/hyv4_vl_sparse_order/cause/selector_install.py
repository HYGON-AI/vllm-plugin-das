import torch
import vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse as m
worker._fix_selector=params['selector']
worker._fix_topk_records=[]
if not hasattr(worker,'_fix_original_topk'):
    worker._fix_original_topk=m._lightop_topk_indices_decode
# Always replace the closure so a source update cannot silently miss the worker.
def select(logits,seq_lens,next_n,topk_indices,topk_tokens):
    ends=m._decode_row_ends_from_seq_lens(seq_lens,next_n,logits.shape[0])
    selector=worker._fix_selector
    if selector=='stable':
        cols=torch.arange(logits.shape[-1],device=logits.device)
        valid=cols.unsqueeze(0)<ends.unsqueeze(1)
        scores=logits.masked_fill(~valid,float('-inf'))
        selected=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:topk_tokens].to(torch.int32)
        seq=cols[:topk_tokens].to(torch.int32).expand_as(selected)
        selected=torch.where((ends<=topk_tokens).unsqueeze(1),torch.where(seq<ends.unsqueeze(1),seq,-1),selected)
        topk_indices.copy_(selected)
    else:
        worker._fix_original_topk(logits,seq_lens,next_n,topk_indices,topk_tokens)
        if selector=='canonical':
            ordered=topk_indices.sort(dim=-1).values
            topk_indices.copy_(torch.where((ends>topk_tokens).unsqueeze(1),ordered,topk_indices))
    if worker._ready_trace and int(ends.max().item())==2049:
        worker._fix_topk_records.append({'logits':logits.detach().cpu().clone(),'indices':topk_indices.detach().cpu().clone(),'ends':ends.cpu().clone(),'selector':selector})
m._lightop_topk_indices_decode=select
result={'rank':worker.rank,'selector':worker._fix_selector,'wrapper_version':2,'canonical_branch_installed':True}
