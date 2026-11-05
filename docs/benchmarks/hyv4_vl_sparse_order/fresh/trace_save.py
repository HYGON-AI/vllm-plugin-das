import torch
if worker.rank==0:torch.save(worker._fresh_records,params['path']+'.logits.pt')
torch.save(worker._fresh_indices,params['path']+f'.indices.rank{worker.rank}.pt')
result={'rank':worker.rank,'records':len(worker._fresh_indices)}
