import ast
from pathlib import Path
import vllm_hcu.models.hy_v4.attention as a
import vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse as m
from vllm_hcu.models.hy_v4.sparse_indices import canonicalize_topk_indices
from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
m._lightop_topk_indices_decode=worker._fix_original_topk
HYV4FlashMLASparseImpl._bf16_flash_mla_kernel_with_lse=worker._fix_original_attention
a.canonicalize_topk_indices=canonicalize_topk_indices
path=Path('/tmp/vllm-plugin-das-pr/vllm_hcu/models/hy_v4/attention.py')
tree=ast.parse(path.read_text())
cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='Indexer')
fn=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='forward')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),a.__dict__,ns)
a.Indexer.forward=ns['forward']
count=0
for mod in worker.model_runner.model.modules():
    if isinstance(mod,a.Indexer):
        mod.canonicalize_topk=True;count+=1
result={'rank':worker.rank,'indexers_updated':count,'source_forward_installed':True}
