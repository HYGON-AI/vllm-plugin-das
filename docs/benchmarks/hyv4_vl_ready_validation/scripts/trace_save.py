import torch
if worker.rank==0:
    torch.save(worker._ready_records,params['path'])
result={'rank':worker.rank,'records':len(worker._ready_records)}
