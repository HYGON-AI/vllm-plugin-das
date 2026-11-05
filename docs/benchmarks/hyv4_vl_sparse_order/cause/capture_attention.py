import torch
from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
if not hasattr(worker,'_fix_original_attention'):
    worker._fix_original_attention=HYV4FlashMLASparseImpl._bf16_flash_mla_kernel_with_lse
worker._fix_attention=None
def attention(impl,q,kv,indices,topk_length=None):
    out=worker._fix_original_attention(impl,q,kv,indices,topk_length)
    if worker.rank==0 and worker._ready_trace and len(worker._fix_topk_records)==1 and worker._fix_attention is None:
        idx=indices.reshape(-1).long()
        selected=kv.reshape(-1,1,kv.shape[-1])[idx.clamp_min(0)]
        worker._fix_attention={'q':q.detach().cpu().clone(),'kv':selected.detach().cpu().clone(),'indices':indices.detach().cpu().clone(),'out':out[0].detach().cpu().clone(),'sinks':None if impl.sinks is None else impl.sinks.detach().cpu().clone(),'scale':impl.softmax_scale,'topk_length':None if topk_length is None else topk_length.detach().cpu().clone()}
    return out
HYV4FlashMLASparseImpl._bf16_flash_mla_kernel_with_lse=attention
result={'rank':worker.rank,'installed':True}
