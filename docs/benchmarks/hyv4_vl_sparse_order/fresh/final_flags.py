from vllm_hcu.models.hy_v4.attention import Indexer
indexers=[m for m in worker.model_runner.model.modules() if isinstance(m,Indexer)]
result={"rank":worker.rank,"enabled_indexers":sum(m.canonicalize_topk for m in indexers),"indexers":len(indexers),"mode":worker._stage4_mode}
