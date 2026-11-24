"""Model policy tests; kernel, Graph and distributed validation are separate."""
from importlib import import_module
from types import SimpleNamespace as NS

import pytest
import torch


def owned():
    return import_module('hcu_vllm_das.hy4_fused_routing')


def config():
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4',hidden_size=6144,
        n_routed_experts=256,num_experts_per_tok=8),dtype=torch.bfloat16,is_multimodal_model=False),
        lora_config=None,use_v2_model_runner=True,compilation_config=NS(mode=0),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            prefill_context_parallel_size=1,decode_context_parallel_size=1,
            data_parallel_size=16,enable_expert_parallel=True,enable_eplb=False,enable_elastic_ep=False))


def tree():
    from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    layer=HYV4MoEFused.__new__(HYV4MoEFused);torch.nn.Module.__init__(layer)
    layer.n_routed_experts=256
    layer.expert_bias=torch.nn.Parameter(torch.zeros(256,dtype=torch.float32))
    router=GroupedTopKRouter(top_k=8,global_num_experts=256,num_expert_group=1,topk_group=1,
        renormalize=True,scoring_func='sigmoid',routed_scaling_factor=2.827,
        e_score_correction_bias=layer.expert_bias)
    layer.experts=NS(router=router,moe_config=NS(moe_parallel_config=NS(
        all2all_backend='deepep_low_latency',ep_size=16,tp_size=1,dp_size=16,pcp_size=1)))
    return torch.nn.ModuleList([layer]),layer,router


@pytest.mark.parametrize('master,child,expected',[(None,None,False),('1',None,False),
    ('0','1',False),('1','0',False),('1','1',True)])
def test_explicit_master_and_child(monkeypatch,master,child,expected):
    m=owned();monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    for name,value in [('VLLM_HCU_USE_CUSTOM_OPS',master),('HCU_VLLM_HY4_FUSED_ROUTING',child)]:
        if value is None:monkeypatch.delenv(name,raising=False)
        else:monkeypatch.setenv(name,value)
    assert m.enabled(config()) is expected


@pytest.mark.parametrize('change',['model','multimodal','dtype','lora','runner','compile',
    'tp','pp','pcp','dcp','dp','no_ep','eplb','elastic','hidden','experts','topk','arch'])
def test_initial_scope_refuses_untested_execution(monkeypatch,change):
    m=owned();cfg=config();monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    if change=='model':cfg.model_config.hf_config.model_type='other'
    elif change=='multimodal':cfg.model_config.is_multimodal_model=True
    elif change=='dtype':cfg.model_config.dtype=torch.float16
    elif change=='lora':cfg.lora_config=object()
    elif change=='runner':cfg.use_v2_model_runner=False
    elif change=='compile':cfg.compilation_config.mode=3
    elif change=='hidden':cfg.model_config.hf_config.hidden_size=1024
    elif change=='experts':cfg.model_config.hf_config.n_routed_experts=128
    elif change=='topk':cfg.model_config.hf_config.num_experts_per_tok=4
    elif change=='arch':monkeypatch.setattr(m,'_device_arch',lambda:'gfx936')
    else:
        fields={'tp':'tensor_parallel_size','pp':'pipeline_parallel_size',
            'pcp':'prefill_context_parallel_size','dcp':'decode_context_parallel_size','dp':'data_parallel_size',
            'no_ep':'enable_expert_parallel','eplb':'enable_eplb','elastic':'enable_elastic_ep'}
        values={'tp':2,'pp':2,'pcp':16,'dcp':2,'dp':8,'no_ep':False,'eplb':True,'elastic':True}
        setattr(cfg.parallel_config,fields[change],values[change])
    assert not m._supported_config(cfg)


def test_supported_config(monkeypatch):
    m=owned();monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    assert m._supported_config(config())


@pytest.mark.parametrize('mode',['dbo','ubatch','batch_invariant','aiter'])
def test_unvalidated_runner_modes_keep_native_routing(monkeypatch,mode):
    m=owned();cfg=config();monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    if mode=='dbo':cfg.parallel_config.enable_dbo=True
    elif mode=='ubatch':cfg.parallel_config.ubatch_size=2
    elif mode=='batch_invariant':
        from vllm import envs
        monkeypatch.setattr(envs,'VLLM_BATCH_INVARIANT',True)
    else:
        from vllm._aiter_ops import rocm_aiter_ops
        monkeypatch.setattr(rocm_aiter_ops,'is_fused_moe_enabled',lambda:True)
    assert not m._supported_config(cfg)


@pytest.mark.parametrize('ubatch_size',[0,1])
def test_native_no_ubatching_defaults_remain_admitted(monkeypatch,ubatch_size):
    m=owned();cfg=config();monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    cfg.parallel_config.enable_dbo=False;cfg.parallel_config.ubatch_size=ubatch_size
    assert m._supported_config(cfg)


def test_disabled_attachment_preserves_every_identity(monkeypatch):
    m=owned();root,layer,router=tree();before=router._compute_routing.__func__
    monkeypatch.setattr(m,'enabled',lambda cfg:False)
    assert m.attach(root,config()) is root
    assert router._compute_routing.__func__ is before
    assert not hasattr(router,'_hcu_native_grouped_compute')


def test_owned_instance_reuses_real_core_select_and_state(monkeypatch):
    m=owned();root,layer,router=tree()
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    original_class_method=GroupedTopKRouter._compute_routing
    original_select=router.select_experts.__func__
    original_state=set(root.state_dict());original_bias=router.e_score_correction_bias
    original_type=router.routing_method_type;captured=[];router.set_capture_fn(captured.append)
    monkeypatch.setattr(m,'enabled',lambda cfg:True)
    monkeypatch.setattr(m,'_requested',lambda:True)
    monkeypatch.setattr(m,'_can_route',lambda *args:True)
    calls=[];weights=torch.ones((2,8),dtype=torch.float32);ids=torch.arange(8,dtype=torch.int64).expand(2,-1)
    monkeypatch.setattr(m,'fused_topk_bias',lambda **kw:calls.append(kw) or (weights,ids))
    assert m.attach(root,config()) is root
    bound=router._compute_routing.__func__;assert bound is not original_class_method
    assert GroupedTopKRouter._compute_routing is original_class_method
    assert router.select_experts.__func__ is original_select
    assert router.routing_method_type==original_type and router.e_score_correction_bias is original_bias
    assert set(root.state_dict())==original_state
    assert m.attach(root,config()) is root and router._compute_routing.__func__ is bound
    hidden=torch.ones((2,6144),dtype=torch.bfloat16);logits=torch.zeros((2,256),dtype=torch.float32)
    actual=router.select_experts(hidden,logits,torch.int64)
    assert actual[0] is weights and torch.equal(actual[1],ids)
    assert calls[0]['hidden_states'] is hidden and calls[0]['gating_output'] is logits
    assert calls[0]['e_score_correction_bias'].data_ptr()==original_bias.data_ptr()
    assert calls[0]['indices_type']==torch.int64 and calls[0]['routed_scaling_factor']==2.827
    assert calls[0]['scoring_func']=='sigmoid' and calls[0]['renormalize'] is True
    assert len(captured)==1 and torch.equal(captured[0],ids)


@pytest.mark.parametrize('condition',['master_off','metadata'])
def test_call_time_fallback_reuses_original(monkeypatch,condition):
    m=owned();root,layer,router=tree()
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    sentinel=object();seen=[]
    monkeypatch.setattr(GroupedTopKRouter,'_compute_routing',lambda self,*a,**kw:seen.append((a,kw)) or sentinel)
    monkeypatch.setattr(m,'enabled',lambda cfg:True)
    m.attach(root,config())
    monkeypatch.setattr(m,'_requested',lambda:condition!='master_off')
    monkeypatch.setattr(m,'_can_route',lambda *args:False)
    result=router._compute_routing('hidden','logits',torch.int64,input_ids='tokens')
    assert result is sentinel and seen==[(('hidden','logits',torch.int64),{'input_ids':'tokens'})]


@pytest.mark.parametrize('change',['logit_dtype','hidden_dtype','bias_dtype','logit_stride','bias_stride',
    'empty','rows','hidden_width','logit_width','bias_width','no_bias','topk','groups','selected_groups',
    'renormalize','score','shared_slots','dtype_request','nan_scale','zero_scale'])
def test_metadata_contract_has_no_device_value_scan(monkeypatch,change):
    m=owned();root,layer,router=tree();h=torch.ones((2,6144),dtype=torch.bfloat16)
    x=torch.ones((2,256),dtype=torch.float32);kind=torch.int32
    monkeypatch.setattr(m,'_device_supported',lambda device:True)
    if change=='logit_dtype':x=x.half()
    elif change=='hidden_dtype':h=h.float()
    elif change=='bias_dtype':router.e_score_correction_bias=router.e_score_correction_bias.half()
    elif change=='logit_stride':x=torch.ones(2,512)[:,::2]
    elif change=='bias_stride':router.e_score_correction_bias=torch.ones(512)[::2]
    elif change=='empty':x=x[:0];h=h[:0]
    elif change=='rows':x=x[:1]
    elif change=='hidden_width':h=h[:,:1024]
    elif change=='logit_width':x=x[:,:128].contiguous()
    elif change=='bias_width':router.e_score_correction_bias=torch.ones(128)
    elif change=='no_bias':router.e_score_correction_bias=None
    elif change=='topk':router.top_k=4
    elif change=='groups':router.num_expert_group=2
    elif change=='selected_groups':router.topk_group=2
    elif change=='renormalize':router.renormalize=False
    elif change=='score':router.scoring_func='softmax'
    elif change=='shared_slots':router.num_fused_shared_experts=1
    elif change=='dtype_request':kind=torch.int16
    elif change=='nan_scale':router.routed_scaling_factor=float('nan')
    elif change=='zero_scale':router.routed_scaling_factor=0
    assert not m._can_route(router,h,x,kind)


def test_positive_metadata_and_cpu_fallback(monkeypatch):
    m=owned();root,layer,router=tree();h=torch.ones(2,6144,dtype=torch.bfloat16);x=torch.ones(2,256)
    assert not m._can_route(router,h,x,torch.int32)
    monkeypatch.setattr(m,'_device_supported',lambda device:True)
    for kind in [None,torch.int32,torch.int64]:assert m._can_route(router,h,x,kind)


@pytest.mark.parametrize('kind',['ct','static','mtp'])
def test_factory_attachment_after_native_and_dense(monkeypatch,kind):
    m=owned()
    from hcu_vllm_das.models import hy4_ct as ct,hy4_static as static
    result=object();cfg=config();seen=[]
    monkeypatch.setattr(m,'attach',lambda model,cfg:seen.append('routing') or model)
    monkeypatch.setattr(ct.dense_fusion,'attach',lambda model,cfg:seen.append('dense') or model)
    monkeypatch.setattr(ct.ihc_launch,'selected',lambda cfg:False)
    if kind=='mtp':
        monkeypatch.setattr(ct.mtp,'HYV4MultiTokenPredictor',lambda **kw:seen.append('native') or result)
        factory=ct.make_ct_predictor
    else:
        owner=ct if kind=='ct' else static
        name='HcuHY4CTBackbone' if kind=='ct' else 'HcuHY4StaticBackbone'
        monkeypatch.setattr(owner,name,lambda **kw:seen.append('native') or result)
        monkeypatch.setattr(owner.hy4_dp_metadata,'prepare',lambda cfg:None)
        factory=owner.make_ct_backbone if kind=='ct' else owner.make_static_backbone
    assert factory(vllm_config=cfg,prefix='model') is result
    assert seen==['native','dense','routing']
