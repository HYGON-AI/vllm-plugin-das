"""Experimental one-process TP16 meta construction; no GPU allocations or loading."""
import json, traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig
from hcu_vllm_das.quantization.kimi31_load_model import Kimi31SlimQuantLoadOnly, Kimi31SlimQuantLoadOnlyInner, OUTER_ARCH, INNER_ARCH
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.model_loader import get_model_loader
from vllm.engine.arg_utils import EngineArgs
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.utils.torch_utils import set_default_torch_dtype
import vllm.distributed.parallel_state as ps
root=Path('/public4/share/wufan/model/Kimi-K3.1-Pro-CHANNEL-INT4-w4a8')
out={'scope':'single-process fake TP16 rank 0, 93-layer outer meta construction using explicit public registries; no weights or real distributed group','ok':False}
try:
    register_quantization_config('slimquant_w4a8')(Kimi31SlimQuantLoadOnlyConfig)
    ModelRegistry.register_model(OUTER_ARCH, Kimi31SlimQuantLoadOnly)
    ModelRegistry.register_model(INNER_ARCH, Kimi31SlimQuantLoadOnlyInner)
    text=json.loads((root/'config.json').read_text())['text_config']; text['num_hidden_layers']=93
    text['min_moe_intermediate_per_partition']=192
    cfg=EngineArgs(model=str(root),trust_remote_code=True,hf_overrides={'text_config':text,'architectures':[OUTER_ARCH]},
      tensor_parallel_size=16,load_format='safetensors',language_model_only=False,
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
        with set_default_torch_dtype(cfg.model_config.dtype), torch.device('meta'):
            model=get_model_loader(cfg.load_config).create_model(cfg,cfg.model_config)
    out['outer_class']=type(model).__qualname__
    out['inner_class']=type(model.language_model).__qualname__
    out['vision_blocks']=len(model.vision_tower.encoder.blocks)
    assert out['vision_blocks'] == 27
    assert out['outer_class'] == 'Kimi31SlimQuantLoadOnly'
    assert out['inner_class'] == 'Kimi31SlimQuantLoadOnlyInner'
    model=model.language_model.model
    out['layers']=len(model.layers)
    out['parameters']={name:{'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device)}
                       for name,p in model.named_parameters() if ('layers.0.self_attn.in_proj_qkvgfab' in name or 'layers.1.block_sparse_moe.experts.w13' in name)}
    out['expert_weights']={name:{'shape':list(param.shape),'dtype':str(param.dtype),'device':str(param.device)}
                           for name,param in model.named_parameters() if 'w13_weight' in name or 'w2_weight' in name}
    assert len(model.layers) == 93
    assert len([n for n in out['expert_weights'] if n.endswith('w13_weight')]) == 92
    assert len([n for n in out['expert_weights'] if n.endswith('w2_weight')]) == 92
    assert out['expert_weights'], "no expert quantized weights were created"
    for name, spec in out['expert_weights'].items():
        if name.endswith('w13_weight'):
            assert spec['shape'] == [896,384,1792] and spec['dtype'] == 'torch.int8', (name,spec)
        if name.endswith('w2_weight'):
            assert spec['shape'] == [896,3584,96] and spec['dtype'] == 'torch.int8', (name,spec)
    assert any(name.endswith('w13_weight') for name in out['expert_weights'])
    assert any(name.endswith('w2_weight') for name in out['expert_weights'])
    out['ok']=True
except BaseException:
    out['traceback']=traceback.format_exc()
finally:
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if not out['ok']: raise SystemExit(1)
