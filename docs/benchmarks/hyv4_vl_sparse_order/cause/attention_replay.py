import torch
from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
result={'rank':worker.rank}
if worker.rank==0:
    data=worker._fix_attention
    impl=next(mod.impl for mod in worker.model_runner.model.modules() if isinstance(getattr(mod,'impl',None),HYV4FlashMLASparseImpl))
    assert torch.equal(impl.sinks.cpu(),data['sinks'])
    q=data['q'].cuda();kv=data['kv'].reshape(-1,576).cuda()
    n=kv.shape[0];idx=torch.arange(n,device=q.device,dtype=torch.int32).view(1,-1)
    length=torch.full((1,),n,device=q.device,dtype=torch.int32)
    outputs=[]
    repeat_exact=[]
    for indices in [idx,idx.flip(-1)]:
        previous=None;ok=True
        for _ in range(10):
            out,_=worker._fix_original_attention(impl,q,kv,indices,length)
            out=out.detach().cpu().clone()
            if previous is not None:ok=ok and torch.equal(previous,out)
            previous=out
        outputs.append(previous);repeat_exact.append(ok)
    torch.save({'q':data['q'],'kv':data['kv'],'sinks':data['sinks'],'scale':data['scale'],'outputs':outputs,'same_order_repeat_exact':repeat_exact},params['path'])
    result.update(same_order_repeat_exact=repeat_exact,permutation_max_abs=(outputs[0].float()-outputs[1].float()).abs().max().item(),changed_elements=int((outputs[0]!=outputs[1]).sum()),shape=list(outputs[0].shape))
