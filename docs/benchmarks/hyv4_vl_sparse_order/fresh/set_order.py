from vllm_hcu.models.hy_v4.attention import Indexer
count=0
for mod in worker.model_runner.model.modules():
    if isinstance(mod,Indexer):
        mod.canonicalize_topk=params['enabled'];count+=1
result={'rank':worker.rank,'enabled':params['enabled'],'indexers':count,'note':'Python eager execution only; already captured FULL graphs retain canonical ordering'}
