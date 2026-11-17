"""Experimental one-process TP16 meta construction; no GPU allocations or loading."""
import os
import json, traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig
from hcu_vllm_das.quantization.kimi31_load_model import _Model
from vllm.engine.arg_utils import EngineArgs
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.utils.torch_utils import set_default_torch_dtype
import vllm.distributed.parallel_state as ps
root=Path(os.environ["KIMI_K31_CHECKPOINT"])
out={'scope':'single-process fake TP16 rank 0, 2-layer CPU model, 2 experts, simulated rank 0: selected checkpoint tensors only','ok':False}
try:
    register_quantization_config('slimquant_w4a8')(Kimi31SlimQuantLoadOnlyConfig)
    text=json.loads((root/'config.json').read_text())['text_config']; text['num_hidden_layers']=2; text['num_experts']=2; text['num_experts_per_token']=2
    text['min_moe_intermediate_per_partition']=192
    cfg=EngineArgs(model=str(root),trust_remote_code=True,hf_overrides={'text_config':text},
      tensor_parallel_size=16,load_format='safetensors',language_model_only=True,
      max_model_len=4096,max_num_seqs=2,enforce_eager=True,disable_custom_all_reduce=True,
      kv_cache_dtype='bfloat16',kernel_config=KernelConfig(moe_backend='emulation',enable_flashinfer_autotune=False)).create_engine_config()
    fake=SimpleNamespace(world_size=16,rank_in_group=0,is_first_rank=True,is_last_rank=True,
                         device_group=None,cpu_group=None,rank=0,ranks=list(range(16)))
    ps._TP=fake; one=SimpleNamespace(world_size=1,rank_in_group=0,is_first_rank=True,is_last_rank=True,device_group=None,cpu_group=None,rank=0,ranks=[0]); ps._PP=one; ps._DP=one; ps._EP=one; ps._DCP=one; ps._PCP=one
    import vllm.models.kimi_k3.nvidia.dsa as dsa
    class DiagnosticDSAPlatform:
        def __init__(self, original): self.original = original
        def is_rocm(self): return False
        def __getattr__(self, name): return getattr(self.original, name)
    dsa.current_platform=DiagnosticDSAPlatform(dsa.current_platform)
    out["dsa_guard_bypassed_for_constructor_only"] = True
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("meta"))
    with set_current_vllm_config(cfg):
        with set_default_torch_dtype(cfg.model_config.dtype), torch.device('cpu'):
            model=_Model(vllm_config=cfg,prefix='model')
    from safetensors import safe_open
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    base='language_model.model.'
    kda=['layers.0.self_attn.'+b+'.weight' for b in ('q_proj','k_proj','v_proj','g_proj','f_a_proj','b_proj')]
    kda += ['layers.0.self_attn.'+b+'.weight_scale' for b in ('q_proj','k_proj','v_proj')]
    expert=['layers.1.block_sparse_moe.experts.0.'+b+s for b in ('w1','w2','w3') for s in ('.weight','.weight_scale')]
    names=kda+expert
    tensors={}
    for shard in set(index[base+n] for n in names):
        with safe_open(str(root/shard),framework='pt',device='cpu') as f:
            for n in names:
                if index[base+n]==shard: tensors[n]=f.get_tensor(base+n).clone()
    loaded=model.load_weights(iter((n,tensors[n]) for n in names))
    proj=model.layers[0].self_attn.in_proj_qkvgfab
    for branch in ('q_proj','k_proj','v_proj'):
        for suffix in ('weight','weight_scale'):
            expected=tensors['layers.0.self_attn.'+branch+'.'+suffix].chunk(16,dim=0)[0]
            assert torch.equal(getattr(getattr(proj,branch),suffix),expected), (branch,suffix)
    gates=[tensors['layers.0.self_attn.g_proj.weight'].chunk(16,0)[0],tensors['layers.0.self_attn.f_a_proj.weight'],
           tensors['layers.0.self_attn.b_proj.weight'].chunk(16,0)[0],torch.zeros(10,7168,dtype=torch.bfloat16)]
    assert torch.equal(proj.fp_gate.weight,torch.cat(gates))
    e=model.layers[1].block_sparse_moe.experts.routed_experts
    stem='layers.1.block_sparse_moe.experts.0.'
    assert e.w13_weight.shape == (2,384,1792) and e.w2_weight.shape == (2,3584,96)
    assert torch.equal(e.w13_weight[0,:192],tensors[stem+'w1.weight'][:192])
    assert torch.equal(e.w13_weight[0,192:],tensors[stem+'w3.weight'][:192])
    assert torch.equal(e.w2_weight[0],tensors[stem+'w2.weight'][:,:96])
    assert torch.equal(e.w13_weight_scale[0,:192],tensors[stem+'w1.weight_scale'][:192])
    assert torch.equal(e.w13_weight_scale[0,192:],tensors[stem+'w3.weight_scale'][:192])
    assert torch.equal(e.w2_weight_scale[0],tensors[stem+'w2.weight_scale'])
    out['loaded_parameter_count']=len(loaded); out['selected_checkpoint_tensors']=len(names)
    out['loaded_parameter_names']=sorted(loaded); out['expert_shape']=list(e.w13_weight.shape)
    out['ok']=True
except BaseException:
    out['traceback']=traceback.format_exc()
finally:
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if not out['ok']: raise SystemExit(1)
