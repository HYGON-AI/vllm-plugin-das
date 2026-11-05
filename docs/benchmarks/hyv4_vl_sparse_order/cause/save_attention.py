import torch
if worker.rank==0:
    torch.save(worker._fix_attention,params['path'])
result={'rank':worker.rank,'captured':worker._fix_attention is not None}
