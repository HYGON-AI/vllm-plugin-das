import torch
path=params['path']+f'.rank{worker.rank}.pt'
torch.save(worker._fix_topk_records,path)
result={'rank':worker.rank,'path':path,'records':len(worker._fix_topk_records),'selector':worker._fix_selector}
