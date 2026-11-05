# Original installed closure during the measured selector ablations.
import torch
import vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse as m
worker._ready_topk_records=[]
worker._ready_selector=params.get('selector','original')
if not hasattr(worker,'_ready_topk_original'):
    worker._ready_topk_original=m._lightop_topk_indices_decode
    def probe(logits,seq_lens,next_n,topk_indices,topk_tokens):
        selector=worker._ready_selector
        ends=m._decode_row_ends_from_seq_lens(seq_lens,next_n,logits.shape[0])
        if selector=='torch':
            topk_indices.copy_(m._topk_indices_torch(logits,topk_tokens,row_ends=ends))
        elif selector=='legacy':
            m._get_lightop_attention().top_k_per_row_decode(logits,1,ends.to(torch.int32).contiguous(),topk_indices,logits.shape[0],logits.stride(0),logits.stride(1),topk_tokens)
        else:
            worker._ready_topk_original(logits,seq_lens,next_n,topk_indices,topk_tokens)
        if worker.rank==0 and worker._ready_trace:
            length=int(ends.max().item())
            if length in (2048,2049) and not any(v['length']==length for v in worker._ready_topk_records):
                worker._ready_topk_records.append({'length':length,'selector':selector,'logits':logits.detach().cpu().clone(),'indices':topk_indices.detach().cpu().clone()})
    m._lightop_topk_indices_decode=probe
result={'rank':worker.rank,'selector':worker._ready_selector}
