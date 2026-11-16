# SPDX-License-Identifier: Apache-2.0
"""Static physical-capacity and checkpoint loader through public factory seams."""
import json,os
from pathlib import Path
import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.utils import is_weights_pre_processed
from ..quantization.minimax31 import MiniMax31W4A8MoEMethod
from .minimax31_static_contract import from_current,layer_id,ABI

class StaticMoEConfig(FusedMoEConfig):
    def __init__(self,**kwargs):
        if (kwargs.get('num_experts'),kwargs.get('num_logical_experts'),kwargs.get('num_local_experts'))!=(128,128,16):raise ValueError('static factory expected untouched128-logical linear baseline')
        p=kwargs['moe_parallel_config']
        if p.enable_eplb or (p.tp_size,p.dp_size,p.pcp_size,p.ep_size)!=(1,1,8,8):raise ValueError('static capacity requires TP1/PCP8/EP8 and runtime EPLB disabled')
        kwargs.update(num_experts=136,num_local_experts=17)
        super().__init__(**kwargs)

class StaticExpertMapManager(ExpertMapManager):
    def update(self,*args,**kwargs):raise RuntimeError('static placement is immutable; online expert-map update is forbidden')

def sample(t):
    flat=t.detach().reshape(-1);n=flat.numel();ix=sorted(set(k*(n-1)//7 for k in range(8)));v=flat[torch.tensor(ix,device=t.device)].cpu().tolist()
    return {'shape':list(t.shape),'dtype':str(t.dtype),'flat_indices':ix,'values':v}

def parameter_samples(layer):
    return {name:[sample(getattr(layer,name)[s]) for s in range(17)] for name in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale')}

class StaticMoEMethod(MiniMax31W4A8MoEMethod):
    def process_weights_after_loading(self,layer):
        if layer.local_num_experts!=17 or self.quant_config.routed_backend!='deepgemm_hipc_v1':raise RuntimeError('static experts require17 slots and audited HIPC backend')
        if is_weights_pre_processed():
            if layer._static_load_receipts:raise RuntimeError('IPC restore unexpectedly ran checkpoint loaders')
            super().process_weights_after_loading(layer)
            layer._static_pack_count=0
            return
        if getattr(layer,'_static_pack_count',0)!=0:raise RuntimeError('static permanent weights may be packed only once')
        expected={(slot,name,shard) for slot in range(17) for name,shard in [('w13_weight','w1'),('w13_weight','w3'),('w2_weight','w2'),('w13_weight_scale','w1'),('w13_weight_scale','w3'),('w2_weight_scale','w2')]}
        if set(layer._static_load_receipts)!=expected:raise RuntimeError('static producer is missing or duplicated17-slot weights/scales')
        self._independent_checkpoint_check(layer)
        super().process_weights_after_loading(layer)
        layer._static_pack_count=1
        layer._static_loading_proof={'schema':'minimax31-static-loading-proof-v1','layer':layer._static_layer_id,'rank':layer.expert_map_manager.ep_rank,'map_sha256':layer._static_placement.map_sha256,'loader_abi':ABI,'physical_slots':list(range(layer.expert_map_manager.ep_rank*17,(layer.expert_map_manager.ep_rank+1)*17)),'logical_owners':list(layer._static_local_owners),'permanent_pack_count':1,'raw_full_copy_checks':len(expected),'independent_checkpoint_sample_checks':len(expected),'raw_loading_receipts':[layer._static_load_receipts[k] for k in sorted(expected)],'packed_samples':parameter_samples(layer)}
        target=os.environ.get('HCU_MINIMAX_STATIC_RECEIPTS')
        if not target:raise RuntimeError('static producer requires a loading-receipt directory')
        folder=Path(target);folder.mkdir(parents=True,exist_ok=True);path=folder/('rank%d_layer%02d.json'%(layer.expert_map_manager.ep_rank,layer._static_layer_id))
        with path.open('x') as f:json.dump(layer._static_loading_proof,f,sort_keys=True);f.write('\n')
    @staticmethod
    def _independent_checkpoint_check(layer):
        from safetensors import safe_open
        from vllm.config import get_current_vllm_config
        root=Path(get_current_vllm_config().model_config.model);index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];byfile={}
        for key,rec in layer._static_load_receipts.items():
            source=rec['checkpoint_name']
            if source not in index:raise RuntimeError('independent checkpoint key missing: '+source)
            byfile.setdefault(index[source],[]).append(rec)
        for filename,records in byfile.items():
            with safe_open(str(root/filename),framework='pt',device='cpu') as f:
                for rec in records:
                    if sample(f.get_tensor(rec['checkpoint_name']))!=rec['source_sample']:raise RuntimeError('independent checkpoint sample mismatch: '+rec['checkpoint_name'])

class StaticRoutedExperts(RoutedExperts):
    def __init__(self,layer_name,params_dtype,moe_config,quant_config,expert_map_manager,**kwargs):
        placement=from_current()
        if placement is None or not isinstance(moe_config,StaticMoEConfig):raise ValueError('static expert construction requires explicit immutable contract')
        if (expert_map_manager.global_num_experts,expert_map_manager.local_num_experts)!=(128,16) or expert_map_manager.routing_tables is not None or expert_map_manager.rocm_aiter_enabled:raise ValueError('unexpected original expert manager')
        if quant_config.routed_backend!='deepgemm_hipc_v1':raise ValueError('static capacity is supported only by the HIPC runner')
        self._static_placement=placement;self._static_layer_id=layer_id(layer_name);self._static_load_receipts={};self._static_pack_count=0
        manager=StaticExpertMapManager(max_num_batched_tokens=moe_config.max_num_tokens,top_k=4,global_num_experts=136,num_redundant_experts=8,num_expert_group=None,moe_parallel_config=moe_config.moe_parallel_config,placement_strategy='linear',enable_eplb=False,num_fused_shared_experts=0,rocm_aiter_enabled=False)
        self._static_local_owners=placement.local(self._static_layer_id,manager.ep_rank)
        super().__init__(layer_name,params_dtype,moe_config,quant_config,expert_map_manager=manager,**kwargs)
        self.register_buffer('_static_physical_to_logical',torch.tensor(placement.row(self._static_layer_id),dtype=torch.int32,device='cpu'),persistent=False)
    def _get_quant_method(self,prefix,quant_config,moe_config):
        method=super()._get_quant_method(prefix,quant_config,moe_config)
        if type(method) is not MiniMax31W4A8MoEMethod:raise RuntimeError('unaudited quantization method for static experts')
        return StaticMoEMethod(quant_config,moe_config)
    def update_expert_map(self):raise RuntimeError('static expert map cannot change in an existing engine')
    def weight_loader(self,param,loaded_weight,weight_name,shard_id,expert_id,return_success=False):
        if is_weights_pre_processed():raise RuntimeError('static IPC consumer cannot load or mutate checkpoint weights')
        if type(expert_id) is not int or not 0<=expert_id<128 or shard_id not in ('w1','w2','w3'):raise ValueError('static loader accepts logical checkpoint IDs0..127 only')
        names=[n for n in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale') if getattr(self,n) is param]
        if len(names)!=1:raise ValueError('unexpected static expert parameter')
        name=names[0];isdown=name.startswith('w2_')
        if isdown!=(shard_id=='w2'):raise ValueError('static checkpoint projection/parameter mismatch')
        slots=[s for s,e in enumerate(self._static_local_owners) if e==expert_id]
        if not slots:return False if return_success else None
        inter=self.moe_config.intermediate_size_per_partition
        source_name=self.layer_name.rsplit('.experts',1)[0]+'.experts.%d.%s.%s'%(expert_id,shard_id,'weight_scale' if name.endswith('_scale') else 'weight')
        source_sample=sample(loaded_weight)
        for slot in slots:
            target=param.data[slot]
            if not isdown:target=target[:inter] if shard_id=='w1' else target[inter:]
            if target.shape!=loaded_weight.shape or target.dtype!=loaded_weight.dtype:raise ValueError('static raw checkpoint shape/dtype mismatch')
            key=(slot,name,shard_id)
            if key in self._static_load_receipts:raise RuntimeError('duplicate static checkpoint shard load')
            target.copy_(loaded_weight)
            if not torch.equal(target,loaded_weight.to(device=target.device)):raise RuntimeError('static full raw copy check failed')
            self._static_load_receipts[key]={'local_slot':slot,'physical_slot':self.expert_map_manager.ep_rank*17+slot,'logical_expert':expert_id,'parameter':name,'shard':shard_id,'checkpoint_name':source_name,'source_sample':source_sample,'raw_full_copy_equal':True}
        return True if return_success else None
